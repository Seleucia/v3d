import Queue
import threading
from multiprocessing.pool import ThreadPool
import numpy as np
import theano.tensor as T
dtype = T.config.floatX
import argparse
import helper.config as config
import model.model_provider as model_provider
import helper.dt_utils as du
import helper.utils as u
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import time


def train_rnn(params):
   rng = RandomStreams(seed=1234)
   (db_train,X_train,Y_train,S_Train_list_org,F_list_train,G_list_train,db_test,X_test,Y_test,S_Test_list_org,F_list_test,G_list_test)=du.load_pose(params)
   index_train_list,S_Train_list=du.get_seq_indexes(params,S_Train_list_org)
   index_test_list,S_Test_list=du.get_seq_indexes(params,S_Test_list_org)
   params["len_train"]=len(index_train_list)*params['seq_length']
   params["len_test"]=len(index_test_list)*params['seq_length']
   u.start_log(params)
   batch_size=params['batch_size']
   n_train_batches = len(index_train_list)
   n_train_batches /= batch_size

   n_test_batches = len(index_test_list)
   n_test_batches /= batch_size

   nb_epochs=params['n_epochs']

   print("Batch size: %i, train batch size: %i, test batch size: %i"%(batch_size,n_train_batches,n_test_batches))
   u.log_write("Model build started",params)
   if params['run_mode']==1:
      model= model_provider.get_model_pretrained(params,rng)
      u.log_write("Pretrained model loaded: %s"%(params['mfile']),params)
   else:
     model= model_provider.get_model(params,rng)
   u.log_write("Number of parameters: %s"%(model.n_param),params)
   train_errors = np.ndarray(nb_epochs)
   u.log_write("Training started",params)
   val_counter=0
   best_loss=1000
   for epoch_counter in range(nb_epochs):
      if(epoch_counter>0 and params["sindex"]>-1):
          sindex=params["sindex"]
          if(sindex+2<params["seq_length"]):
              params["sindex"]=sindex+2
          else:
              params["sindex"]=0
              u.log_write("One cycle completed",params)
          u.log_write("Training set loading with %d, sliding"%(params["sindex"]),params)
          params["load_mode"]=4
          (X_train,Y_train,S_Train_list_org,F_list_train,G_list_train)=du.load_pose(params,db_train=db_train)

      index_train_list,S_Train_list=du.get_seq_indexes(params,S_Train_list_org)
      n_train_batches = len(index_train_list)
      n_train_batches /= batch_size

      batch_loss = 0.
      LStateList_t=[np.zeros(shape=(batch_size,params['n_hidden']), dtype=dtype) for i in range(model.numOfLayers*2)] # initial hidden state
      LStateList_pre=[np.zeros(shape=(batch_size,params['n_hidden']), dtype=dtype) for i in range(model.numOfLayers*2)] # initial hidden state
      state_reset_counter_lst=[0 for i in range(batch_size)]
      is_train=1
      for minibatch_index in range(n_train_batches):
          state_reset_counter_lst=[s+1 for s in state_reset_counter_lst]
          (LStateList_b,x,y,state_reset_counter_lst)=du.prepare_lstm_batch(index_train_list, minibatch_index, batch_size, S_Train_list,LStateList_t,LStateList_pre, F_list_train, params, Y_train, X_train,state_reset_counter_lst)
          LStateList_pre=LStateList_b
          args=(x, y,is_train)+tuple(LStateList_b)
          result= model.train(*args)
          loss=result[0]
          LStateList_t=result[1:len(result)]
          batch_loss += loss
      if params['shufle_data']==1:
         X_train,Y_train=du.shuffle_in_unison_inplace(X_train,Y_train)
      train_errors[epoch_counter] = batch_loss
      batch_loss/=n_train_batches
      s='TRAIN--> epoch %i | error %f'%(epoch_counter, batch_loss)
      u.log_write(s,params)
      if(epoch_counter%1==0):
          is_train=0
          print("Model testing")
          batch_loss3d = []
          LStateList_t=[np.zeros(shape=(batch_size,params['n_hidden']), dtype=dtype) for i in range(model.numOfLayers*2)] # initial hidden state
          LStateList_pre=[np.zeros(shape=(batch_size,params['n_hidden']), dtype=dtype) for i in range(model.numOfLayers*2)] # initial hidden sta
          state_reset_counter_lst=[0 for i in range(batch_size)]
          for minibatch_index in range(n_test_batches):
             state_reset_counter_lst=[s+1 for s in state_reset_counter_lst]
             (LStateList_b,x,y,state_reset_counter_lst)=du.prepare_lstm_batch(index_test_list, minibatch_index, batch_size, S_Test_list, LStateList_t,LStateList_pre, F_list_test, params, Y_test, X_test,state_reset_counter_lst)
             LStateList_pre=LStateList_b
             args=(x,is_train)+tuple(LStateList_b)
             result = model.predictions(*args)
             pred=result[0]
             LStateList_t=result[1:len(result)]
             loss3d =u.get_loss(params,y,pred)
             batch_loss3d.append(loss3d)
          batch_loss3d=np.nanmean(batch_loss3d)
          if(batch_loss3d<best_loss):
             best_loss=batch_loss3d
             ext=str(epoch_counter)+"_"+str(batch_loss3d)+"_best.p"
             u.write_params(model.params,params,ext)
          else:
              ext=str(val_counter%2)+".p"
              u.write_params(model.params,params,ext)

          val_counter+=1#0.08
          s ='VAL--> epoch %i | error %f, %f'%(val_counter,batch_loss3d,n_test_batches)
          u.log_write(s,params)


params= config.get_params()
parser = argparse.ArgumentParser(description='Training the module')
parser.add_argument('-m','--model',help='Model: lstm, lstm2, erd current('+params["model"]+')',default=params["model"])
args = vars(parser.parse_args())
params["model"]=args["model"]
params=config.update_params(params)
train_rnn(params)