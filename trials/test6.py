import numpy
import theano
import theano.tensor as T

batch_size=1
params=[]
dtype=numpy.float32

dtype = T.config.floatX

h_t_1=c_t_1=h_t_2=c_t_2=h_t_3=c_t_3=numpy.zeros(shape=(batch_size,512), dtype=dtype) # initial hidden state

h_t_1[0,1]=1

H = T.matrix(name="H",dtype=dtype) # initial hidden state
C = T.matrix(name="H",dtype=dtype) # initial hidden state
C =C*2
updates = []
updates.append((C,C*2))

fn=theano.function(inputs=[H,C],outputs=[H,C],updates=updates,allow_input_downcast=True)
x,y=fn(h_t_1,c_t_1)
print y
# print c_t_1


