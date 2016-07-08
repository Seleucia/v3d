import numpy as np
from multiprocessing.pool import ThreadPool
import helper.config as config
import model.model_provider as model_provider
import helper.dt_utils as du
import helper.utils as u
import plot.plot_utils as pu
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
params= config.get_params()
params["model"]="cnn4"
params['mfile']= "cnn4_cnnX_0_0.185977_best.p"
params["data_dir"]="/mnt/Data1/hc/joints16/"
# params["data_dir"]="/home/coskun/PycharmProjects/data/pose/joints16/"
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

n_test = len(Y)
residual=n_test%batch_size
#residual=0
if residual>0:
   residual=batch_size-residual
   F_list=F_list.tolist()
   Y_List=Y.tolist()
   f=F_list[-1]
   y=Y_List[-1]
   for i in range(residual):
      F_list.append(f)
      Y_List.append(y)
   F_list=np.asarray(F_list)
   Y=np.asarray(Y_List)
   n_test = len(Y)

# n_test_batches =n_test/ batch_size
n_batches = len(Y)
n_batches /= batch_size

print("Sample size: %i, Batch size: %i, #batch: %i"%(len(Y),batch_size,n_batches))
print ("Model loading started")
model= model_provider.get_model_pretrained(params,rng)
print("Prediction started")
batch_loss = 0.
batch_loss3d = []
batch_loss = []
loss_list=[]
last_index=0
first_index=0
sq_loss_lst=[]
for minibatch_index in range(n_batches):
   if(minibatch_index==0):
      x,y=du.prepare_cnn_batch(minibatch_index, batch_size, F_list, Y)
   pool = ThreadPool(processes=2)
   async_t = pool.apply_async(model.predictions, (x,is_train))
   async_b = pool.apply_async(du.prepare_cnn_batch, (minibatch_index, batch_size, F_list, Y))
   pool.close()
   pool.join()
   pred = async_t.get()  # get the return value from your function.
   loss3d =u.get_loss(params,y,pred)
   loss=np.nanmean(np.abs(pred -y))
   batch_loss3d.append(loss3d)
   batch_loss.append(loss)
   x=[]
   y=[]
   (x,y) = async_b.get()  # get the return value from your function.

   if(minibatch_index==n_batches-1):
      pred= model.predictions(x,is_train)
      pred = pred[0:(len(pred)-residual)]
      y=y[0:(len(y)-residual)]
      loss3d =u.get_loss(params,y,pred)
      batch_loss3d.append(loss3d)

#   du.write_predictions(params,pred,n_list)
   #u.write_pred(pred,minibatch_index,G_list,params)

batch_loss=np.mean(batch_loss)
batch_loss3d=np.mean(batch_loss3d)
print "============================================================================"
s ='error %f, 3d error: %f'%(batch_loss,batch_loss3d)
print (s)
#pu.plot_cumsum(loss_list)
