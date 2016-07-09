import glob
import random
import math
from PIL import Image
import os
import numpy
import collections
from multiprocessing.dummy import Pool as ThreadPool
import theano
import theano.tensor as T
from random import randint
dtype = T.config.floatX

def load_pose(params,load_mode=0,only_pose=1,sindex=0):
   data_dir=params["data_dir"]
   max_count=params["max_count"]
   seq_length=params["seq_length"]
   # dataset_reader=read_full_midlayer #read_full_joints,read_full_midlayer
   # dataset_reader=multi_thr_read_full_joints_sequence_cnn #cnn+lstm training data loading
   # dataset_reader=multi_thr_read_full_joints #read_full_joints,read_full_midlayer
   # dataset_reader=multi_thr_read_full_midlayer_sequence #lstm training with autoencoder layer
   # dataset_reader=multi_thr_read_full_joints_sequence #read_full_joints,read_full_midlayer
   # dataset_reader=multi_thr_read_full_midlayer_cnn #read_full_midlayer
   dataset_reader=multi_thr_read_full_joints_cnn #read_full_joints,read_full_midlayer
   # dataset_reader=joints_sequence_tp1 #read_full_joints,read_full_midlayer

   if(load_mode==2):
       mode=2
       get_flist=False
       (X,Y,F_list,G_list,S_list)= dataset_reader(data_dir,max_count,seq_length,sindex,mode,get_flist)
       print "Full data set loaded"
       return (X,Y,F_list,G_list,S_list)

   elif load_mode==1:#load only test
       mode=1
       get_flist=False
       (X_test,Y_test,F_list_test,G_list_test,S_Test_list)= dataset_reader(data_dir,max_count,seq_length,sindex,mode,get_flist)
       print "Test set loaded"
       return (X_test,Y_test,F_list_test,G_list_test,S_Test_list)

   elif load_mode==0:#Load training and testing seperate list
       mode=0
       get_flist=False
       X_train,Y_train,F_list_train,G_list_train,S_Train_list=dataset_reader(data_dir,max_count,seq_length,sindex,mode,get_flist)
       print "Test set loaded"
       mode=1
       (X_test,Y_test,F_list_test,G_list_test,S_Test_list)= dataset_reader(data_dir,max_count,seq_length,sindex,mode,get_flist)
       print "Training set loaded"
       return (X_train,Y_train,S_Train_list,F_list_train,G_list_train,X_test,Y_test,S_Test_list,F_list_test,G_list_test)
   else:
        raise Exception('You should pass mode argument for data loading.!') #


   # if(params["model"]=="cnn"):
   #     X_train=X_train.reshape(X_train.shape[0]*X_train.shape[1],X_train.shape[2])
   #     Y_train=Y_train.reshape(Y_train.shape[0]*Y_train.shape[1],Y_train.shape[2])
   #     X_test=X_test.reshape(X_test.shape[0]*X_test.shape[1],X_test.shape[2])
   #     Y_test=Y_test.reshape(Y_test.shape[0]*Y_test.shape[1],Y_test.shape[2])




def read_full_midlayer_sequence(base_file,max_count,p_count,sindex,istest,get_flist=False):
    f_dir="/mnt/Data1/hc/auto/"
    if istest==0:
        lst_act=['S1','S5','S6','S7','S8']
    else:
        lst_act=['S9','S11']
    X_D=[]
    Y_D=[]
    F_L=[]
    G_L=[]
    S_L=[]
    seq_id=0
    for actor in lst_act:
        tmp_folder=base_file+actor+"/"
        lst_sq=os.listdir(tmp_folder)
        for sq in lst_sq:
            X_d=[]
            Y_d=[]
            F_l=[]
            seq_id+=1
            tmp_folder=base_file+actor+"/"+sq+"/"
            file_list=os.listdir(tmp_folder)
            for fl in file_list:
                p=tmp_folder+fl
                if not os.path.isfile(p):
                   continue
                with open(p, "rb") as f:
                  data=f.read().strip().split(' ')
                  y_d= [float(val) for val in data]
                  Y_d.append(numpy.asarray(y_d)/1000)
                  F_l.append(p)
                p=f_dir+actor+'/'+sq+'/'+fl
                with open(p, "rb") as f:
                  rd=f.read()
                  data=rd.strip().split('\n')
                  x_d= [float(val) for val in data]
                  X_d.append(numpy.asarray(x_d))
                if len(X_d)==p_count and p_count>0:
                        X_D.append(X_d)
                        Y_D.append(Y_d)
                        F_L.append(F_l)
                        S_L.append(seq_id)
                        X_d=[]
                        Y_d=[]
                        F_l=[]
                if len(Y_D)>=max_count:
                    return (numpy.asarray(X_D,dtype=numpy.float32),numpy.asarray(Y_D,dtype=numpy.float32),F_L,G_L,S_L)

    return (numpy.asarray(X_D,dtype=numpy.float32),numpy.asarray(Y_D,dtype=numpy.float32),F_L,G_L,S_L)

def multi_thr_read_full_joints_sequence_cnn(base_file,max_count,p_count,sindex,mode,get_flist=False):
    joints_file=base_file
    img_folder=base_file.replace('joints16','h36m_rgb_img_crop')
    if mode==0:#load training data.
        lst_act=['S1','S5','S6','S7','S8']
    elif mode==1:#load test data
        lst_act=['S9','S11']
    elif mode==2:#load full data
        lst_act=['S1','S5','S6','S7','S8','S9','S11']
    else:
        raise Exception('You should pass mode argument for data loading.!') #
    X_D=[]
    Y_D=[]
    F_L=[]
    G_L=[]
    S_L=[]
    seq_id=0
    for actor in lst_act:
        tmp_folder=base_file+actor+"/"
        lst_sq=os.listdir(tmp_folder)
        for sq in lst_sq:
            X_d=[]
            Y_d=[]
            F_l=[]
            seq_id+=1
            tmp_folder=joints_file+actor+"/"+sq+"/"
            tmp_folder_img=img_folder+actor+"/"+sq.replace('.cdf','')+"/"
            id_list=os.listdir(tmp_folder)
            if os.path.exists(tmp_folder_img)==False:
                continue
            img_count=len(os.listdir(tmp_folder_img))
            min_count=img_count
            if(len(id_list)<img_count):
                min_count=len(id_list)
            if min_count==0:
                continue
            seq_id+=1
            id_list=id_list[0:min_count]
            joint_list=[tmp_folder + p1 for p1 in id_list]
            img_list=[img_folder+actor+'/'+sq.replace('.cdf','')+'/frame_'+(p1.replace('.txt','')).zfill(5)+'.png' for p1 in id_list]
            pool = ThreadPool(1000)
            results = pool.map(load_file, joint_list)
            pool.close()

            for r in range(len(results)):
                rs=results[r]
                f=img_list[r]
                Y_d.append(rs)
                F_l.append(f)
                if len(Y_d)==p_count and p_count>0:
                        Y_D.append(Y_d)
                        F_L.append(F_l)
                        S_L.append(seq_id)
                        Y_d=[]
                        F_l=[]
                if len(Y_D)>=max_count:
                    return (numpy.asarray(X_D,dtype=numpy.float32),numpy.asarray(Y_D,dtype=numpy.float32),numpy.asarray(F_L),G_L,S_L)
        if(len(Y_d)>0):
            residual=len(Y_d)%p_count
            residual=p_count-residual
            y=residual*[Y_d[-1]]
            f=residual*[F_l[-1]]
            Y_d.extend(y)
            F_l.extend(f)
            if len(Y_d)==p_count and p_count>0:
                S_L.append(seq_id)
                Y_D.append(Y_d)
                F_L.append(F_l)
                Y_d=[]
                F_l=[]
                if len(Y_D)>=max_count:
                    return (numpy.asarray(X_D,dtype=numpy.float32),numpy.asarray(Y_D,dtype=numpy.float32),numpy.asarray(F_L),G_L,S_L)


    return (numpy.asarray(X_D,dtype=numpy.float32),numpy.asarray(Y_D,dtype=numpy.float32),numpy.asarray(F_L),G_L,S_L)

def multi_thr_read_full_joints_cnn(base_file,max_count,p_count,sindex,mode,get_flist=False):
    #CNN training with only autoencoder layer
    joints_file=base_file
    img_folder=base_file.replace('joints16','h36m_rgb_img_crop')
    if mode==0:#load training data.
        lst_act=['S1','S5','S6','S7','S8']
    elif mode==1:#load test data
        lst_act=['S9','S11']
    elif mode==2:#load full data
        lst_act=['S11','S1','S5','S6','S7','S8','S9']
    else:
        raise Exception('You should pass mode argument for data loading.!') #
    X_D=[]
    Y_D=[]
    F_L=[]
    G_L=[]
    S_L=[]
    seq_id=0
    for actor in lst_act:
        tmp_folder=joints_file+actor+"/"
        lst_sq=os.listdir(tmp_folder)
        if(max_count>-1):
            avg=2000.
            cnt=math.ceil(float(max_count)/(float(len(lst_act))*avg))
            lst_sq=random(lst_act,cnt)

        for sq in lst_sq:
            # if 'Greeting' not in sq:
            #     continue
            # X_d=[]
            # Y_d=[]
            # F_l=[]
            print sq
            tmp_folder=joints_file+actor+"/"+sq+"/"
            tmp_folder_img=img_folder+actor+"/"+sq.replace('.cdf','')+"/"
            id_list=os.listdir(tmp_folder)
            if os.path.exists(tmp_folder_img)==False:
                continue
            img_count=len(os.listdir(tmp_folder_img))
            min_count=img_count
            if(len(id_list)<img_count):
                min_count=len(id_list)
            if min_count==0:
                continue
            seq_id+=1
            id_list=id_list[0:min_count]
            joint_list=[tmp_folder + p1 for p1 in id_list]
            midlayer_list=[img_folder+actor+'/'+sq.replace('.cdf','')+'/frame_'+(p1.replace('.txt','')).zfill(5)+'.png' for p1 in id_list]
            pool = ThreadPool(1000)
            results = pool.map(load_file, joint_list)
            pool.close()
            Y_D.extend(results)
            F_L.extend(midlayer_list)
            if len(Y_D)>=max_count:
                Y_D=numpy.asarray(Y_D,dtype=numpy.float32)
                F_L=numpy.asarray(F_L)
                return (X_D,Y_D,F_L,G_L,S_L)

    Y_D=numpy.asarray(Y_D,dtype=numpy.float32)
    F_L=numpy.asarray(F_L)
    return (X_D,Y_D,F_L,G_L,S_L)

def multi_thr_read_full_midlayer_cnn(base_file,max_count,p_count,sindex,istest,get_flist=False):
    #CNN training with only autoencoder layer
    joints_file=base_file
    img_folder=base_file.replace('auto','h36m_rgb_img_crop')
    if istest==0:
        lst_act=['S1','S5','S6','S7','S8']
    else:
        lst_act=['S9','S11']
    X_D=[]
    Y_D=[]
    F_L=[]
    G_L=[]
    S_L=[]
    seq_id=0
    for actor in lst_act:
        tmp_folder=joints_file+actor+"/"
        lst_sq=os.listdir(tmp_folder)
        for sq in lst_sq:
            # X_d=[]
            # Y_d=[]
            # F_l=[]
            tmp_folder=joints_file+actor+"/"+sq+"/"
            tmp_folder_img=img_folder+actor+"/"+sq.replace('.cdf','')+"/"
            id_list=os.listdir(tmp_folder)
            if os.path.exists(tmp_folder_img)==False:
                continue
            img_count=len(os.listdir(tmp_folder_img))
            min_count=img_count
            if(len(id_list)<img_count):
                min_count=len(id_list)
            if min_count==0:
                continue
            seq_id+=1
            id_list=id_list[0:min_count]
            joint_list=[tmp_folder + p1 for p1 in id_list]
            midlayer_list=[img_folder+actor+'/'+sq.replace('.cdf','')+'/frame_'+(p1.replace('.txt','')).zfill(5)+'.png' for p1 in id_list]
            pool = ThreadPool(1000)
            results = pool.map(load_file_nodiv, joint_list)
            pool.close()
            Y_D.extend(results)
            F_L.extend(midlayer_list)
            if len(Y_D)>=max_count:
                Y_D=numpy.asarray(Y_D,dtype=numpy.float32)
                F_L=numpy.asarray(F_L)
                return (X_D,Y_D,F_L,G_L,S_L)

    Y_D=numpy.asarray(Y_D,dtype=numpy.float32)
    F_L=numpy.asarray(F_L)
    return (X_D,Y_D,F_L,G_L,S_L)

def multi_thr_read_full_joints_sequence(base_file,max_count,p_count,sindex,istest,get_flist=False):
    #LSTM training with only joints
    base_file=base_file.replace('img','joints16')
    if istest==0:
        lst_act=['S1','S5','S6','S7','S8']
    else:
        lst_act=['S9','S11']
    X_D=[]
    Y_D=[]
    F_L=[]
    G_L=[]
    S_L=[]
    seq_id=0
    for actor in lst_act:
        tmp_folder=base_file+actor+"/"
        lst_sq=os.listdir(tmp_folder)
        for sq in lst_sq:
            X_d=[]
            Y_d=[]
            F_l=[]
            seq_id+=1
            tmp_folder=base_file+actor+"/"+sq+"/"
            id_list=os.listdir(tmp_folder)
            joint_list=[tmp_folder + p1 for p1 in id_list]
            pool = ThreadPool(1000)
            results = pool.map(load_file, joint_list)
            pool.close()
            for r in range(len(results)):
                rs=results[r]
                Y_d.append(rs)
                if len(Y_d)==p_count and p_count>0:
                        Y_D.append(Y_d)
                        S_L.append(seq_id)
                        Y_d=[]
                        F_l=[]
                if len(Y_D)>=max_count:
                    X_D=Y_D=numpy.asarray(Y_D,dtype=numpy.float32)
                    return (X_D,Y_D,F_L,G_L,S_L)
        if(len(Y_d)>0):
            residual=len(Y_d)%p_count
            residual=p_count-residual
            y=residual*[Y_d[-1]]
            # f=residual*[F_l[-1]]
            Y_d.extend(y)
            # F_l.extend(f)
            if len(Y_d)==p_count and p_count>0:
                S_L.append(seq_id)
                Y_D.append(Y_d)
                # F_L.append(F_l)
                Y_d=[]
                # F_l=[]
                if len(Y_D)>=max_count:
                    X_D=Y_D=numpy.asarray(Y_D,dtype=numpy.float32)
                    return (X_D,Y_D,F_L,G_L,S_L)


    X_D=Y_D=numpy.asarray(Y_D,dtype=numpy.float32)
    return (X_D,Y_D,F_L,G_L,S_L)

def joints_sequence_tp1(base_file,max_count,p_count,sindex,mode,get_flist=False):
    #LSTM training with only joints
    if mode==0:#load training data.
        lst_act=['S1','S5','S6','S7','S8']
    elif mode==1:#load test data
        lst_act=['S9','S11']
    elif mode==2:#load full data
        lst_act=['S1','S5','S6','S7','S8','S9','S11']
    else:
        raise Exception('You should pass mode argument for data loading.!') #
    X_D=[]
    Y_D=[]
    F_L=[]
    G_L=[]
    S_L=[]
    seq_id=0
    for actor in lst_act:
        tmp_folder=base_file+actor+"/"
        lst_sq=os.listdir(tmp_folder)
        for sq in lst_sq:
            X_d=[]
            Y_d=[]
            F_l=[]
            seq_id+=1
            tmp_folder=base_file+actor+"/"+sq+"/"
            id_list=os.listdir(tmp_folder)
            joint_list=[tmp_folder + p1 for p1 in id_list]
            pool = ThreadPool(1000)
            results = pool.map(load_file, joint_list)
            pool.close()
            for r in range(len(results)-1):
                rs=results[r]
                X_d.append(rs)
                rs_1=results[r+1]
                Y_d.append(rs_1)
                if len(Y_d)==p_count and p_count>0:
                        Y_D.append(Y_d)
                        X_D.append(X_d)
                        S_L.append(seq_id)
                        Y_d=[]
                        X_d=[]
                        F_l=[]
                if len(Y_D)>=max_count:
                    X_D=numpy.asarray(X_D,dtype=numpy.float32)
                    Y_D=numpy.asarray(Y_D,dtype=numpy.float32)
                    return (X_D,Y_D,F_L,G_L,S_L)
        if(len(Y_d)>0):
            residual=len(Y_d)%p_count
            residual=p_count-residual
            y=residual*[Y_d[-1]]
            x=residual*[X_d[-1]]
            # f=residual*[F_l[-1]]
            Y_d.extend(y)
            X_d.extend(x)
            if len(Y_d)==p_count and p_count>0:
                S_L.append(seq_id)
                Y_D.append(Y_d)
                X_D.append(X_d)
                # F_L.append(F_l)
                Y_d=[]
                X_d=[]
                # F_l=[]
                if len(Y_D)>=max_count:
                    X_D=numpy.asarray(X_D,dtype=numpy.float32)
                    Y_D=numpy.asarray(Y_D,dtype=numpy.float32)
                    return (X_D,Y_D,F_L,G_L,S_L)


    X_D=numpy.asarray(X_D,dtype=numpy.float32)
    Y_D=numpy.asarray(Y_D,dtype=numpy.float32)
    return (X_D,Y_D,F_L,G_L,S_L)

def multi_thr_read_full_midlayer_sequence(base_file,max_count,p_count,sindex,istest,get_flist=False):
    f_dir="/mnt/hc/auto/"
    if istest==0:
        lst_act=['S1','S5','S6','S7','S8']
    else:
        lst_act=['S9','S11']
    X_D=[]
    Y_D=[]
    F_L=[]
    G_L=[]
    S_L=[]
    seq_id=0
    for actor in lst_act:
        tmp_folder=base_file+actor+"/"
        lst_sq=os.listdir(tmp_folder)
        for sq in lst_sq:
            X_d=[]
            Y_d=[]
            F_l=[]
            seq_id+=1
            tmp_folder=base_file+actor+"/"+sq+"/"
            id_list=os.listdir(tmp_folder)
            joint_list=[tmp_folder + p1 for p1 in id_list]
            midlayer_list=[f_dir+actor+'/'+sq+'/'+p1 for p1 in id_list]
            pool = ThreadPool(1000)
            results = pool.map(load_file, joint_list)
            pool.close()

            for r in range(len(results)):
                rs=results[r]
                f=midlayer_list[r]
                Y_d.append(rs)
                F_l.append(f)
                if len(Y_d)==p_count and p_count>0:
                        Y_D.append(Y_d)
                        F_L.append(F_l)
                        S_L.append(seq_id)
                        Y_d=[]
                        F_l=[]
                if len(Y_D)>=max_count:
                    return (numpy.asarray(X_D,dtype=numpy.float32),numpy.asarray(Y_D,dtype=numpy.float32),numpy.asarray(F_L),G_L,S_L)
        if(len(Y_d)>0):
            residual=len(Y_d)%p_count
            residual=p_count-residual
            y=residual*[Y_d[-1]]
            f=residual*[F_l[-1]]
            Y_d.extend(y)
            F_l.extend(f)
            if len(Y_d)==p_count and p_count>0:
                S_L.append(seq_id)
                Y_D.append(Y_d)
                F_L.append(F_l)
                Y_d=[]
                F_l=[]
                if len(Y_D)>=max_count:
                    return (numpy.asarray(X_D,dtype=numpy.float32),numpy.asarray(Y_D,dtype=numpy.float32),numpy.asarray(F_L),G_L,S_L)


    return (numpy.asarray(X_D,dtype=numpy.float32),numpy.asarray(Y_D,dtype=numpy.float32),numpy.asarray(F_L),G_L,S_L)

def read_full_midlayer(base_file,max_count,p_count,sindex,istest,get_flist=False):
    base_file=base_file.replace('img','joints')

    f_dir="/mnt/Data1/hc/auto/"
    if istest==0:
        lst_act=['S1','S5','S6','S7','S8']
    else:
        lst_act=['S9','S11','S1','S5','S6','S7','S8']
    X_D=[]
    Y_D=[]
    F_L=[]
    G_L=[]
    S_L=[]
    seq_id=0
    for actor in lst_act:
        tmp_folder=base_file+actor+"/"
        lst_sq=os.listdir(tmp_folder)
        for sq in lst_sq:
            seq_id+=1
            tmp_folder=base_file+actor+"/"+sq+"/"
            file_list=os.listdir(tmp_folder)
            for fl in file_list:
                p=tmp_folder+fl
                if not os.path.isfile(p):
                   continue
                with open(p, "rb") as f:
                  data=f.read().strip().split(' ')
                  y_d= [float(val) for val in data]
                  Y_D.append(numpy.asarray(y_d)/1000)
                  F_L.append(p)
                p=f_dir+actor+'/'+sq+'/'+fl
                with open(p, "rb") as f:
                  rd=f.read()
                  data=rd.strip().split('\n')
                  x_d= [float(val) for val in data]
                  X_D.append(numpy.asarray(x_d))



            if len(Y_D)>=max_count:
                X_D=Y_D
                return (numpy.asarray(X_D,dtype=numpy.float32),numpy.asarray(Y_D,dtype=numpy.float32),F_L,G_L,S_L)

    X_D=Y_D
    return (numpy.asarray(X_D,dtype=numpy.float32),numpy.asarray(Y_D,dtype=numpy.float32),F_L,G_L,S_L)

def read_full_joints(base_file,max_count,p_count,sindex,istest,get_flist=False):
    base_file=base_file.replace('img','joints16')
    if istest==0:
        lst_act=['S1','S5','S6','S7','S8']        
    else:
        lst_act=['S9','S11']
    X_D=[]
    Y_D=[]
    F_L=[]
    G_L=[]
    S_L=[]
    seq_id=0
    for actor in lst_act:
        tmp_folder=base_file+actor+"/"
        lst_sq=os.listdir(tmp_folder)
        for sq in lst_sq:
            seq_id+=1
            tmp_folder=base_file+actor+"/"+sq+"/"
            file_list=os.listdir(tmp_folder)
            # print file_list
            for fl in file_list:
                p=tmp_folder+fl
                if not os.path.isfile(p):
                   continue
                with open(p, "rb") as f:
                  data=f.read().strip().split(' ')
                  y_d= [float(val) for val in data]
                  Y_D.append(numpy.asarray(y_d)/1000)
                  if p in F_L:
                      print p
                  F_L.append(p)

            if len(Y_D)>=max_count:
                X_D=Y_D
                return (numpy.asarray(X_D,dtype=numpy.float32),numpy.asarray(Y_D,dtype=numpy.float32),F_L,G_L,S_L)

    X_D=Y_D
    return (numpy.asarray(X_D,dtype=numpy.float32),numpy.asarray(Y_D,dtype=numpy.float32),F_L,G_L,S_L)

def multi_thr_read_full_joints(base_file,max_count,p_count,sindex,istest,get_flist=False):
    base_file=base_file.replace('img','joints16')
    if istest==0:
        lst_act=['S1','S5','S6','S7','S8']        
    else:
        lst_act=['S9','S11']
        # lst_act=['S1','S5','S6','S7','S8','S9','S11']
    X_D=[]
    Y_D=[]
    F_L=[]
    G_L=[]
    S_L=[]
    seq_id=0
    for actor in lst_act:
        tmp_folder=base_file+actor+"/"
        lst_sq=os.listdir(tmp_folder)
        for sq in lst_sq:
            seq_id+=1
            tmp_folder=base_file+actor+"/"+sq+"/"
            file_list=os.listdir(tmp_folder)
            my_list=[tmp_folder + p1 for p1 in file_list]
            pool = ThreadPool(1000)
            results = pool.map(load_file, my_list)
            Y_D.extend(results)
            F_L.extend(my_list)
            pool.close()
            # print file_list
            #for fl in file_list:
            #    p=tmp_folder+fl
            #    if not os.path.isfile(p):
            #       continue
            #    with open(p, "rb") as f:
            #      data=f.read().strip().split(' ')
            #      y_d= [float(val) for val in data]
            #      Y_D.append(numpy.asarray(y_d)/1000)
            #      if p in F_L:
            #          print p
            #      F_L.append(p)

            if len(Y_D)>=max_count:
                X_D=Y_D
                return (numpy.asarray(X_D,dtype=numpy.float32),numpy.asarray(Y_D,dtype=numpy.float32),F_L,G_L,S_L)

    X_D=Y_D
    return (numpy.asarray(X_D,dtype=numpy.float32),numpy.asarray(Y_D,dtype=numpy.float32),F_L,G_L,S_L)

def load_batch(params,x_lst,y_lst):
    X_d=[]
    Y_d=[]
    base_file=params["data_dir"]
    for f in range(params['batch_size']):
        fl=base_file+x_lst[f]+'.txt'
        gl=base_file.replace('img','joints')+y_lst[f]+'.txt'
        if not os.path.isfile(fl):
           continue
        with open(gl, "rb") as f:
          data=f.read().strip().split(' ')
          y_d= [float(val) for val in data]
          # if(numpy.isnan(numpy.sum(y_d))):
          #     continue;
          Y_d.append(numpy.asarray(y_d)/1000)

        with open(fl, "rb") as f:
           data=f.read().strip().split(' ')
           data=data[0].split('\n')
           x_d = [float(val) for val in data]
           X_d.append(numpy.asarray(x_d)/10)
    return numpy.asarray(X_d),numpy.asarray(Y_d)

def multi_thr_load_batch(my_list):
    lst=my_list[0]
    pool = ThreadPool(len(lst))
    results = pool.map(load_file_nodiv, lst)
    pool.close()
    x=[]
    x.append(results)
    return numpy.asarray(x)

def multi_thr_load_cnn_lstm_batch(my_list):
    lst=my_list
    pool = ThreadPool(len(lst))
    results = pool.map(load_file_cnn_lstm_patch, lst)
    pool.close()
    return numpy.asarray(results)


def multi_thr_load_cnn_batch(my_list):
    lst=my_list
    pool = ThreadPool(len(lst))
    results = pool.map(load_file_patch, lst)
    pool.close()
    return numpy.asarray(results)

def load_file(fl):
    with open(fl, "rb") as f:
        data=f.read().strip().split(' ')
        y_d= [numpy.float32(val) for val in data]
        y_d=numpy.asarray(y_d,dtype=numpy.float32)/1000
        f.close()
        return y_d

def load_file_nodiv(fl):
    with open(fl, "rb") as f:
        rd=f.read()
        data=rd.strip().split('\n')
        x_d= [numpy.float32(val) for val in data]
        x_d=numpy.asarray(x_d,dtype=numpy.float32)
        f.close()
        return x_d

def load_file_patch(fl):
    patch_margin=(0,0)
    orijinal_size=(128,128)
    size=(112,112)
    x1=randint(patch_margin[0],orijinal_size[0]-(patch_margin[0]+size[0]))
    x2=x1+size[0]
    y1=randint(patch_margin[1],orijinal_size[1]-(patch_margin[1]+size[1]))
    y2=y1+size[1]
    normalizer=255
    patch_loc= (x1,y1,x2,y2)
    img = Image.open(fl)
    img = img.crop(patch_loc)
    arr=numpy.asarray(img)
    arr.flags.writeable = True
    arr=arr-(130.70753799,84.31474484,72.801691)
    arr/=normalizer
    arr=numpy.squeeze(arr)
    return arr


def load_file_cnn_lstm_patch(fl):
    patch_margin=(0,0)
    orijinal_size=(128,128)
    size=(112,112)
    x1=randint(patch_margin[0],orijinal_size[0]-(patch_margin[0]+size[0]))
    x2=x1+size[0]
    y1=randint(patch_margin[1],orijinal_size[1]-(patch_margin[1]+size[1]))
    y2=y1+size[1]
    normalizer=255
    patch_loc= (x1,y1,x2,y2)
    img = Image.open(fl)
    img = img.crop(patch_loc)
    arr=numpy.asarray(img)
    arr.flags.writeable = True
    arr/=normalizer
    arr=numpy.squeeze(arr)
    arr=arr.reshape(3*112*112)
    return arr


def prepare_cnn_lstm_batch(index_train_list, minibatch_index, batch_size, S_Train_list, sid, H, C, F_list, params, Y, X):
    id_lst=index_train_list[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024
    tmp_sid=S_Train_list[(minibatch_index + 1) * batch_size-1]
    if(sid==0):
      sid=tmp_sid
    if(tmp_sid!=sid):
      sid=tmp_sid
      H=C=numpy.zeros(shape=(batch_size,params['n_hidden']), dtype=dtype) # resetting initial state, since seq change
    x_fl=F_list[id_lst][0]
    result=multi_thr_load_cnn_lstm_batch(my_list=x_fl)
    x_lst=[]
    x_lst.append(result)
    x=numpy.asarray(x_lst)
    y=Y[id_lst]
    return (sid,H,C,x,y)

def prepare_lstm_batch(index_list, minibatch_index, batch_size, S_list,LStateList, F_list, params, Y, X,state_reset_counter):
    id_lst=index_list[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024
    pre_sid=S_list[(minibatch_index) * batch_size-1]
    curr_sid=S_list[(minibatch_index + 1) * batch_size-1]
    if(pre_sid!=curr_sid) or ((state_reset_counter%params['reset_state']==0) and state_reset_counter*params['reset_state']>0):
      state_reset_counter=0
      new_list=[numpy.zeros(shape=(batch_size,params['n_hidden']), dtype=dtype) for i in range(params['nlayer'])*2] # initial hidden state
    else:
        new_list=LStateList
    if params['mtype']=="seq":
        x=X[id_lst]
    else:
        x_fl=[F_list[f] for f in id_lst]
        x=multi_thr_load_batch(my_list=x_fl)
    y=Y[id_lst]
    return (new_list,x,y,state_reset_counter)


def prepare_lstm_3layer_batch(index_train_list, minibatch_index, batch_size, S_Train_list, sid, h_t_1,c_t_1,h_t_2,c_t_2,h_t_3,c_t_3, F_list, params, Y_train, X_train):
    id_lst=index_train_list[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024
    tmp_sid=S_Train_list[(minibatch_index + 1) * batch_size-1]
    if(sid==0):
      sid=tmp_sid
    if(tmp_sid!=sid):
      sid=tmp_sid
      h_t_1=c_t_1=h_t_2=c_t_2=h_t_3=c_t_3=numpy.zeros(shape=(batch_size,params['n_hidden']), dtype=dtype) # initial hidden state
    if params['model']=='lstm_3layer_joints':
        x=X_train[id_lst]
    else:
        x_fl=F_list[id_lst]
        x=multi_thr_load_cnn_lstm_batch(my_list=x_fl)
    y=Y_train[id_lst]
    return (sid,h_t_1,c_t_1,h_t_2,c_t_2,h_t_3,c_t_3,x,y)

def prepare_cnn_batch(minibatch_index, batch_size, F_list, Y):
    id_lst=range(minibatch_index * batch_size, (minibatch_index + 1) * batch_size,1)
    x_fl=F_list[id_lst].tolist()
    x=multi_thr_load_cnn_batch(my_list=x_fl)
    y=Y[id_lst]
    return (x,y)


def get_batch_indexes(params,S_list):
   SID_List=[]
   counter=collections.Counter(S_list)
   grb_count=counter.values()
   s_id=0
   index_list=range(0,len(S_list),1)
   for mx in grb_count:
       SID_List.extend(numpy.repeat(s_id,mx))
       s_id+=1
   return (index_list,SID_List)

def write_predictions(params,pred,N_list):
   wd=params["wd"]
   base_dir=wd+"/pred/res/"
   counter=0
   for b in pred:
      for i in b:
         path=base_dir+params["model"]+"_"+os.path.split(N_list[counter])[1].replace(".txt","")
         numpy.save(path, i)
         counter=counter+1

def shuffle_in_unison_inplace(a, b):
   assert len(a) == len(b)
   p = numpy.random.permutation(len(a))
   return a[p],b[p]
