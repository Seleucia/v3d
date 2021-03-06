import os
import utils
import platform

def get_params():
   global params
   params={}
   params['run_mode']=0 #0,full,1:resume, 2 = combine models

   params["rn_id"]="lstm" #running id, model
   params["load_mode"]=0#0=full training,1=only test set,2=full dataset, 3=hyper param searc
   params["notes"]="LSTM training with batches and new loading..." #running id
   params["model"]="lstm"#kccnr,dccnr
   params["optimizer"]="Adam" #1=classic kcnnr, 2=patch, 3=conv, 4 =single channcel
   params['mfile']=""
   # params['mfile']= "cnn_1_0.p,lstm_auto_lstm_21_0.0078104_best.p"
   # params['mfile']= "cnn_1_1_24.6707_best2.p,autoencoder_auto_lr_low_138_0.00340542_best.p"

   params['lr']=0.0001
   params['mtype']="seq"
   params['shufle_data']=0
   params['nlayer']= 1 #LSTM
   params['batch_size']=5
   params['seq_length']= 50
   params['reset_state']= 10#-1=Never, n=every n batch
   params["corruption_level"]=0.5

   #system settings
   wd=os.path.dirname(os.path.realpath(__file__))
   wd=os.path.dirname(wd)
   params['wd']=wd
   params['log_file']=wd+"/logs/"+params["model"]+"_"+params["rn_id"]+"_"+str(params['run_mode'])+"_"+utils.get_time()+".txt"
   params["model_file"]=wd+"/cp/"

   # early-stopping parameters
   params['patience']= 10000  # look as this many examples regardless
   params['patience_increase']=2  # wait this much longer when a new best is
   params['improvement_threshold']=0.995  # a relative improvement of this much is

   params["sindex"]=0#can be used for data shiftin
   # learning parameters
   params['momentum']=0.9    # the params for momentum
   params['learning_rate_decay']= 0.998
   params['squared_filter_length_limit']=15.0
   params['n_epochs']=25600
   params['n_hidden']= 1024
   params['n_output']= 48

   if(platform.node()=="coskunh"):
       params["caffe"]="/home/coskun/sftpkg/caffe/python"
       params["data_dir"]="/home/coskun/PycharmProjects/data/pose/joints16/"
       params['n_hidden']= 2
       params['batch_size']=50
       params['max_count']= 2000000

   if(platform.node()=="milletari-workstation"):
       params["data_dir"]="/mnt/Data1/hc/joints16/" #joints with 16, cnn+lstm and autoencder training
       # params["data_dir"]="/mnt/Data1/hc/auto/" #cnn and lstm seperate training must be this
       # params["caffe"]="/usr/local/caffe/python"
       params['max_count']=-1

   if(platform.node()=="titanx2"):
       params["data_dir"]="/home/users/achilles/human36/joints16/" #joints with 16, cnn+lstm and autoencder training
       # params["data_dir"]="/mnt/Data1/hc/auto/" #cnn and lstm seperate training must be this
       # params["caffe"]="/usr/local/caffe/python"
       params['max_count']=-1

   if(platform.node()=="FedeWSLinux"):
       params["caffe"]="/usr/local/caffe/python"
       params["data_dir"]="/mnt/hc/joints16/"
       params['max_count']=1000

   if(platform.node()=="cmp-comp"):
       params['batch_size']=60
       params["n_procc"]=1
       params["WITH_GPU"]=True
       params["caffe"]="/home/coskun/sftpkg/caffe/python"
       params["data_dir"]="/mnt/Data1/hc/joints/"
       params["data_dir"]="/home/huseyin/data/joints16/"
       params['n_hidden']= 128
       params['max_count']= 100


   #params['step_size']=[10]
   params['test_size']=0.20 #Test size
   params['val_size']=0.20 #val size
   params['test_freq']=100 #Test frequency
   return params

def update_params(params):
   params['log_file']=params["wd"]+"/logs/"+params["model"]+"_"+params["rn_id"]+"_"+str(params['run_mode'])+"_"+utils.get_time()+".txt"
   return params
