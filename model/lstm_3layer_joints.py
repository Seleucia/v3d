import numpy as np
import theano
from layers import LSTMLayer,DropoutLayer
import theano.tensor as T
from theano import shared
from helper.utils import init_weight,init_bias,get_err_fn,count_params
from helper.optimizer import RMSprop

dtype = T.config.floatX

class lstm_3layer_joints:
   def __init__(self,rng, params,cost_function='mse',optimizer = RMSprop):
       batch_size=params['batch_size']
       sequence_length=params["seq_length"]

       lr=params['lr']
       self.n_in = 48
       self.n_lstm = params['n_hidden']
       self.n_out = params['n_output']

       X = T.tensor3() # batch of sequence of vector
       Y = T.tensor3() # batch of sequence of vector
       is_train = T.iscalar('is_train') # pseudo boolean for switching between training and prediction

       self.W_hy = init_weight((self.n_lstm, self.n_out), rng=rng,name='W_hy', sample= 'glorot')
       self.b_y = init_bias(self.n_out,rng=rng, sample='zero')

       layer1=LSTMLayer(rng,0,self.n_in,self.n_lstm)
       layer2=LSTMLayer(rng,1,self.n_lstm,self.n_lstm)
       layer3=LSTMLayer(rng,2,self.n_lstm,self.n_lstm)

       self.params = layer1.params+layer2.params+layer3.params
       self.params.append(self.W_hy)
       self.params.append(self.b_y)

       def step_lstm(x_t,mask,h_tm1_1,c_tm1_1,h_tm1_2,c_tm1_2,h_tm1_3,c_tm1_3):
           [h_t_1,c_t_1,y_t_1]=layer1.run(x_t,h_tm1_1,c_tm1_1)
           dl1=DropoutLayer(rng,input=y_t_1,prob=0.5,is_train=is_train,mask=mask)
           [h_t_2,c_t_2,y_t_2]=layer2.run(dl1.output,h_tm1_2,c_tm1_2)
           [h_t_3,c_t_3,y_t_3]=layer3.run(y_t_2,h_tm1_3,c_tm1_3)
           y = T.dot(y_t_3, self.W_hy) + self.b_y
           return [h_t_1,c_t_1,h_t_2,c_t_2,h_t_3,c_t_3,y]

       h0_1 = T.matrix(name="h0_1",dtype=dtype) # initial hidden state
       c0_1 = T.matrix(name="c0_1",dtype=dtype) # initial hidden state
       h0_2 = T.matrix(name="h0_2",dtype=dtype) # initial hidden state
       c0_2 = T.matrix(name="c0_2",dtype=dtype) # initial hidden state
       h0_3 = T.matrix(name="h0_3",dtype=dtype) # initial hidden state
       c0_3 = T.matrix(name="c0_3",dtype=dtype) # initial hidden state

       mask_shape=(sequence_length,batch_size,self.n_lstm)
       p_1=0.5
       mask= rng.binomial(size=mask_shape, p=p_1, dtype=X.dtype)

       noise= rng.normal(size=(batch_size,sequence_length,self.n_in), std=0.008, avg=0.0,dtype=theano.config.floatX)
       X_train=noise+X
       X_tilde= T.switch(T.neq(is_train, 0), X_train, X)

       [h_t_1,c_t_1,h_t_2,c_t_2,h_t_3,c_t_3,y_vals], _ = theano.scan(fn=step_lstm,
                                         sequences=[X_tilde.dimshuffle(1,0,2),mask],
                                         outputs_info=[h0_1, c0_1,h0_2, c0_2, h0_3, c0_3, None])

       self.output = y_vals.dimshuffle(1,0,2)
       cost=get_err_fn(self,cost_function,Y)

       _optimizer = optimizer(
            cost,
            self.params,
            lr=lr
        )

       self.train = theano.function(inputs=[X,Y,is_train,h0_1, c0_1,h0_2, c0_2, h0_3, c0_3],outputs=[cost,h_t_1[-1],c_t_1[-1],h_t_2[-1],c_t_2[-1],h_t_3[-1],c_t_3[-1]],updates=_optimizer.getUpdates(),allow_input_downcast=True)
       self.predictions = theano.function(inputs = [X,is_train,h0_1, c0_1,h0_2, c0_2, h0_3, c0_3], outputs = [self.output,h_t_1[-1],c_t_1[-1],h_t_2[-1],c_t_2[-1],h_t_3[-1],c_t_3[-1]],allow_input_downcast=True)
       self.n_param=count_params(self.params)
