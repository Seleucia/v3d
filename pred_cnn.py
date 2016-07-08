import numpy as np
import helper.config as config
import model.model_provider as model_provider
import helper.dt_utils as du
import helper.utils as u
import plot.plot_utils as pu
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
params= config.get_params()
params["model"]="cnn4"
params['mfile']= "autoenconder_auto_0.p"
params["data_dir"]="/mnt/Data1/hc/joints16/"
# params["data_dir"]="/home/coskun/PycharmProjects/data/rnn/180k/"
#0, error 0.087360, 0.177598, 20.323595
#error 0.078438, 0.161955, 16.453038
#VAL--> epoch 21 | error 0.086701, 0.179906
rng = RandomStreams(seed=1234)
only_test=1
is_train=0
params['seq_length']= 1
params['batch_size']=10
batch_size=params['batch_size']
params['max_count']=100

u.prep_pred_file(params)
sindex=0
(X,Y,F_list,G_list,S_list)=du.load_pose(params,load_mode=2,sindex=0)

n_test = len(X)
residual=n_test%batch_size
#residual=0
if residual>0:
   residual=batch_size-residual
   X_List=X.tolist()
   Y_List=Y.tolist()
   x=X_List[-1]
   y=Y_List[-1]
   for i in range(residual):
      X_List.append(x)
      Y_List.append(y)
   X=np.asarray(X_List)
   Y=np.asarray(Y_List)
   n_test = len(Y)

# n_test_batches =n_test/ batch_size
n_test_batches = len(X)
n_test_batches /= batch_size

print("Test sample size: %i, Batch size: %i, test batch size: %i"%(X_test.shape[0]*X_test.shape[1],batch_size,n_test_batches))
print ("Model loading started")
model= model_provider.get_model_pretrained(params,rng)
print("Prediction started")
batch_loss = 0.
batch_loss3d = 0.
batch_bb_loss = 0.
loss_list=[]
last_index=0
first_index=0
sq_loss_lst=[]
for minibatch_index in range(n_test_batches):
   x=X[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
   y=Y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
   pred = model.predictions(x,is_train)
   # print("Prediction done....")
   if residual>0:
      if(minibatch_index==n_test_batches-1):
         pred = pred[0:(len(pred)-residual)]
         y=y[0:(len(y)-residual)]

#   du.write_predictions(params,pred,n_list)
   #u.write_pred(pred,minibatch_index,G_list,params)
   loss=np.nanmean(np.abs(pred -y))*2
   loss3d =u.get_loss(params,y,pred)
   #print(s_list)
   batch_loss += loss
   batch_loss3d += loss3d
batch_loss/=n_test_batches
batch_loss3d/=n_test_batches
print "============================================================================"
print sq_loss_lst
s ='error %f, %f, %f,%f'%(batch_loss,batch_loss3d,n_test_batches)
print (s)
#pu.plot_cumsum(loss_list)
