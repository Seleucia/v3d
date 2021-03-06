import theano
from layers import LSTMLayer,MDNoutputLayer
import theano.tensor as T
from helper.utils import init_weight,init_bias,count_params,nll, do_nothing
from helper.optimizer import RMSprop
dtype = T.config.floatX

class lstm_mdn:
   def __init__(self,rng, params,cost_function='mse',optimizer = RMSprop):
       batch_size=params['batch_size']
       sequence_length=params["seq_length"]

       lr=params['lr']
       self.n_in = 48
       self.n_lstm = params['n_hidden']
       self.n_out = params['n_output']
       n_fc=512

       self.W_hy = init_weight((self.n_lstm, n_fc), rng=rng,name='W_hy', sample= 'glorot')
       self.b_y = init_bias(n_fc,rng=rng, sample='zero')

       self.numOfLayers=1
       layer1=LSTMLayer(rng,0,self.n_in,self.n_lstm)

       self.params = layer1.params
       self.params.append(self.W_hy)
       self.params.append(self.b_y)

       def step_lstm(x_t,h_tm1_1,c_tm1_1):
           [h_t_1,c_t_1,y_t_1]=layer1.run(x_t,h_tm1_1,c_tm1_1)
           y = T.dot(y_t_1, self.W_hy) + self.b_y
           return [h_t_1,c_t_1,y]

       X = T.tensor3() # batch of sequence of vector
       Y = T.tensor3() # batch of sequence of vector
       is_train = T.iscalar('is_train') # pseudo boolean for switching between training and prediction
       H = T.matrix(name="H",dtype=dtype) # initial hidden state
       C = T.matrix(name="C",dtype=dtype) # initial hidden state

       noise= rng.normal(size=(batch_size,sequence_length,self.n_in), std=0.008, avg=0.0,dtype=theano.config.floatX)
       X_train=noise+X

       X_tilde= T.switch(T.neq(is_train, 0), X_train, X)

       [h_t_1,c_t_1,y_vals], _ = theano.scan(fn=step_lstm,
                                         sequences=[X_tilde.dimshuffle(1,0,2)],
                                         outputs_info=[H, C, None])

       # self.output = y_vals.dimshuffle(1,0,2)
       mdn_input=y_vals.dimshuffle(1,0,2)
       mdn_input=T.reshape(mdn_input,(batch_size*sequence_length,n_fc))
       Y_ll=T.reshape(Y,(batch_size*sequence_length,params['n_output']))

       mdn =  MDNoutputLayer(rng=rng,
                                           input=mdn_input,
                                           n_in=n_fc,
                                           n_out=params['n_output'],
                                           mu_activation=do_nothing,
                                         n_components=240)
       self.params=self.params+mdn.params
       # self.params.append(mdn.W_mixing)
       # self.params.append(mdn.W_mu)
       # self.params.append(mdn.W_sigma)

       cost = nll(mu = mdn.mu,
                 sigma = mdn.sigma,
                 mixing = mdn.mixing,
                 y = Y_ll) #+ L2_reg * self.frame_pred.L2_sqr

       L2_reg=0.0001
       L2_sqr = theano.shared(0.)
       for param in self.params:
           L2_sqr += (T.sum(param[0] ** 2)+T.sum(param[1] ** 2))

       cost += L2_reg*L2_sqr


       _optimizer = optimizer(
            cost,
            self.params,
            lr=lr
        )

       #Sampling from the GMM
       component = rng.multinomial(pvals=mdn.mixing)
       component_mean =  T.sum(mdn.mu * component.dimshuffle(0,'x',1),axis=2)
       component_std = T.sum(mdn.sigma * component, axis=1, keepdims=True)
       samples=rng.normal(size=(batch_size*sequence_length,params['n_output']),avg = component_mean, std=component_std)

       self.output = T.reshape(samples,(batch_size,sequence_length,params['n_output']))

       self.train = theano.function(inputs=[X,Y,is_train,H,C],outputs=[cost,h_t_1[-1],c_t_1[-1]],updates=_optimizer.getUpdates(),allow_input_downcast=True)
       self.predictions = theano.function(inputs = [X,is_train,H,C], outputs = [self.output,h_t_1[-1],c_t_1[-1]],allow_input_downcast=True)
       self.n_param=count_params(self.params)
