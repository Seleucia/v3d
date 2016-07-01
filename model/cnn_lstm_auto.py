from theano import shared
import numpy as np
import theano
import theano.tensor as T
from layers import ConvLayer,PoolLayer,HiddenLayer,DropoutLayer,LSTMLayer,LogisticRegression
import theano.tensor.nnet as nn
from helper.utils import init_weight,init_bias,get_err_fn,count_params, do_nothing
from helper.optimizer import RMSprop

# theano.config.exception_verbosity="high"
dtype = T.config.floatX

class cnn_lstm_auto(object):
    def __init__(self,rng,params,cost_function='mse',optimizer = RMSprop):

        lr=params["lr"]
        # n_lstm=params['n_hidden']
        # n_out=params['n_output']
        batch_size=params["batch_size"]
        sequence_length=params["seq_length"]

        # minibatch)
        X = T.tensor3() # batch of sequence of vector
        Y = T.tensor3() # batch of sequence of vector
        is_train = T.iscalar('is_train') # pseudo boolean for switching between training and prediction

        #CNN global parameters.
        subsample=(1,1)
        p_1=0.5
        border_mode="same"
        cnn_batch_size=batch_size*sequence_length
        pool_size=(2,2)

        #Layer1: conv2+pool+drop
        filter_shape=(64,3,9,9)
        input_shape=(cnn_batch_size,3,112,112) #input_shape= (samples, channels, rows, cols)
        input=X.reshape(input_shape)
        # input= X_r.dimshuffle(0,3,1,2)
        c1=ConvLayer(rng, input,filter_shape, input_shape,border_mode,subsample, activation=nn.relu)
        p1=PoolLayer(c1.output,pool_size=pool_size,input_shape=c1.output_shape)
        dl1=DropoutLayer(rng,input=p1.output,prob=p_1,is_train=is_train)

        #Layer2: conv2+pool
        filter_shape=(128,p1.output_shape[1],3,3)
        c2=ConvLayer(rng, dl1.output, filter_shape,p1.output_shape,border_mode,subsample, activation=nn.relu)
        p2=PoolLayer(c2.output,pool_size=pool_size,input_shape=c2.output_shape)

        #Layer3: conv2+pool
        filter_shape=(128,p2.output_shape[1],3,3)
        c3=ConvLayer(rng, p2.output,filter_shape,p2.output_shape,border_mode,subsample, activation=nn.relu)
        p3=PoolLayer(c3.output,pool_size=pool_size,input_shape=c3.output_shape)


        #Layer4: conv2+pool
        filter_shape=(64,p3.output_shape[1],3,3)
        c4=ConvLayer(rng, p3.output,filter_shape,p3.output_shape,border_mode,subsample, activation=nn.relu)
        p4=PoolLayer(c4.output,pool_size=pool_size,input_shape=c4.output_shape)

        #Layer5: hidden
        n_in= reduce(lambda x, y: x*y, p4.output_shape[1:])
        x_flat = p4.output.flatten(2)
        h1=HiddenLayer(rng,x_flat,n_in,1024,activation=nn.relu)

        #Layer6: Regressin layer
        lreg=LogisticRegression(rng,h1.output,1024,2048)

        #LSTM paramaters
        self.n_in = 2048
        self.n_lstm = params['n_hidden']
        self.n_out = params['n_output']

        self.W_hy = init_weight((self.n_lstm, self.n_out), rng=rng,name='W_hy', sample= 'glorot')
        self.b_y = init_bias(self.n_out,rng=rng, sample='zero')

        layer1=LSTMLayer(rng,0,self.n_in,self.n_lstm)
        self.params =c1.params+c2.params+c3.params+c4.params+h1.params+lreg.params+layer1.params
        self.params.append(self.W_hy)
        self.params.append(self.b_y)

        def step_lstm(x_t,h_tm1_1,c_tm1_1):
           [h_t_1,c_t_1,y_t_1]=layer1.run(x_t,h_tm1_1,c_tm1_1)
           y = T.dot(y_t_1, self.W_hy) + self.b_y
           return [h_t_1,c_t_1,y]

        H = T.matrix(name="H",dtype=dtype) # initial hidden state
        C = T.matrix(name="C",dtype=dtype) # initial hidden state

        rnn_input = lreg.y_pred.reshape((batch_size,sequence_length, self.n_in))


        [h_t_1,c_t_1,y_vals], _ = theano.scan(fn=step_lstm,
                                         sequences=[rnn_input.dimshuffle(1,0,2)],
                                         outputs_info=[H, C, None])

        self.output = y_vals.dimshuffle(1,0,2)
        cost=get_err_fn(self,cost_function,Y)

        _optimizer = optimizer(
            cost,
            self.params,
            lr=lr
        )

        self.train = theano.function(inputs=[X,Y,is_train,H,C],outputs=[cost,h_t_1[-1],c_t_1[-1]],updates=_optimizer.getUpdates(),allow_input_downcast=True)
        self.predictions = theano.function(inputs = [X,is_train,H,C], outputs = [self.output,h_t_1[-1],c_t_1[-1]],allow_input_downcast=True)
        self.n_param=count_params(self.params)