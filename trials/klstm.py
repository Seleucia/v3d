import theano.tensor as T

a,b = T.dmatrices('a','b')
x,y = T.dmatrices('x','y')

is_train=1

#1=training,2=test
z= T.switch(T.neq(is_train, 0), 1, 2)

print z.eval()

