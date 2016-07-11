import glob
import random
from random import shuffle
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

def load_pose(params,db_train=dict(),db_test=dict(),):
   data_dir=params["data_dir"]
   max_count=params["max_count"]
   seq_length=params["seq_length"]
   load_mode=params["load_mode"]
   sindex=params["sindex"]
   get_flist=False
   # dataset_reader=read_full_midlayer #read_full_joints,read_full_midlayer
   # dataset_reader=multi_thr_read_full_joints_sequence_cnn #cnn+lstm training data loading
   # dataset_reader=multi_thr_read_full_joints #read_full_joints,read_full_midlayer
   # dataset_reader=multi_thr_read_full_midlayer_sequence #lstm training with autoencoder layer
   # dataset_reader=multi_thr_read_full_joints_sequence #read_full_joints,read_full_midlayer
   # dataset_reader=multi_thr_read_full_midlayer_cnn #read_full_midlayer
   # dataset_reader=multi_thr_read_full_joints_cnn #read_full_joints,read_full_midlayer
   dataset_reader=joints_sequence_tp1_v2 #LSTM reading with dataset
   # dataset_reader=joints_sequence_tp12 #read_full_joints,read_full_midlayer

   if load_mode==4:#only trainings
       mode=0
       db_train,X_train,Y_train,F_list_train,G_list_train,S_Train_list=dataset_reader(db_train,data_dir,max_count,seq_length,sindex,mode,get_flist)
       print "Training set loaded"
       return (X_train,Y_train,S_Train_list,F_list_train,G_list_train)
   elif load_mode==3:#Load parameter search
       mode=3
       X_train,Y_train,F_list_train,G_list_train,S_Train_list=dataset_reader(data_dir,max_count,seq_length,sindex,mode,get_flist)
       print "Training set loaded"
       mode=4
       sindex=0
       (X_test,Y_test,F_list_test,G_list_test,S_Test_list)= dataset_reader(data_dir,max_count,seq_length,sindex,mode,get_flist)
       print "Test set loaded"
       return (X_train,Y_train,S_Train_list,F_list_train,G_list_train,X_test,Y_test,S_Test_list,F_list_test,G_list_test)
   elif(load_mode==2):
       mode=2
       sindex=0
       (X,Y,F_list,G_list,S_list)= dataset_reader(data_dir,max_count,seq_length,sindex,mode,get_flist)
       print "Full data set loaded"
       return (X,Y,F_list,G_list,S_list)
   elif load_mode==1:#load only test
       mode=1
       sindex=0
       (X_test,Y_test,F_list_test,G_list_test,S_Test_list)= dataset_reader(data_dir,max_count,seq_length,sindex,mode,get_flist)
       print "Test set loaded"
       return (X_test,Y_test,F_list_test,G_list_test,S_Test_list)
   elif load_mode==0:#Load training and testing seperate list
       mode=0
       db_train,X_train,Y_train,F_list_train,G_list_train,S_Train_list=dataset_reader(db_train,data_dir,max_count,seq_length,sindex,mode,get_flist)
       print "Training set loaded"
       mode=1
       sindex=0
       (db_test,X_test,Y_test,F_list_test,G_list_test,S_Test_list)= dataset_reader(db_test,data_dir,max_count,seq_length,sindex,mode,get_flist)
       print "Test set loaded"
       return (db_train,X_train,Y_train,S_Train_list,F_list_train,G_list_train,db_test,X_test,Y_test,S_Test_list,F_list_test,G_list_test)
   else:
        raise Exception('You should pass mode argument for data loading.!') #


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
        total=1544050.
    elif mode==1:#load test data
        lst_act=['S9','S11']
        total=544700.
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
        for sq in lst_sq:
            # if 'Greeting' not in sq:
            #     continue
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
            if(max_count>-1):
                assert (total>max_count)
                divider=total/max_count
                cnt=int(math.ceil(float(min_count)/divider))
                id_list=random.sample(id_list,cnt)

            joint_list=[tmp_folder + p1 for p1 in id_list]
            midlayer_list=[img_folder+actor+'/'+sq.replace('.cdf','')+'/frame_'+(p1.replace('.txt','')).zfill(5)+'.png' for p1 in id_list]
            pool = ThreadPool(1000)
            results = pool.map(load_file, joint_list)
            pool.close()
            Y_D.extend(results)
            F_L.extend(midlayer_list)
            if len(Y_D)>=max_count and max_count>-1:
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
    elif mode==3:#load full data
        lst_act=['S1',]
    elif mode==4:#load full data
        lst_act=['S11']
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
            # if 'Greeting' not in sq:
            #     continue
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
            sift=1
            for r in range(len(results)-(sift+sindex)):
                t_r=r+sindex
                rs=results[t_r]
                X_d.append(rs)
                rs_1=results[t_r+sift]
                Y_d.append(rs_1)
                if len(Y_d)==p_count and p_count>0:
                        Y_D.append(Y_d)
                        X_D.append(X_d)
                        S_L.append(seq_id)
                        Y_d=[]
                        X_d=[]
                        F_l=[]
                if len(Y_D)>=max_count and max_count>0:
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
                if len(Y_D)>=max_count and max_count>0:
                    X_D=numpy.asarray(X_D,dtype=numpy.float32)
                    Y_D=numpy.asarray(Y_D,dtype=numpy.float32)
                    return (X_D,Y_D,F_L,G_L,S_L)


    X_D=numpy.asarray(X_D,dtype=numpy.float32)
    Y_D=numpy.asarray(Y_D,dtype=numpy.float32)
    return (X_D,Y_D,F_L,G_L,S_L)

def joints_sequence_tp1_v2(db,base_file,max_count,p_count,sindex,mode,get_flist=False):
    #LSTM training with only joints
    if mode==0:#load training data.
        lst_act=['S1','S5','S6','S7','S8']
    elif mode==1:#load test data
        lst_act=['S9','S11']
    elif mode==2:#load full data
        lst_act=['S1','S5','S6','S7','S8','S9','S11']
    elif mode==3:#load full data
        lst_act=['S1',]
    elif mode==4:#load full data
        lst_act=['S11']
    else:
        raise Exception('You should pass mode argument for data loading.!') #
    lst_act=['S11']
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
            # if 'Greeting' not in sq:
            #     continue
            X_d=[]
            Y_d=[]
            F_l=[]
            seq_id+=1
            tmp_folder=base_file+actor+"/"+sq+"/"
            if tmp_folder not in db:
                id_list=os.listdir(tmp_folder)
                joint_list=[tmp_folder + p1 for p1 in id_list]
                pool = ThreadPool(1000)
                results = pool.map(load_file, joint_list)
                pool.close()
                db[tmp_folder]=results
            else:
                results=db[tmp_folder]
            sift=1
            for r in range(len(results)-(sift+sindex)):
                t_r=r+sindex
                rs=results[t_r]
                X_d.append(rs)
                rs_1=results[t_r+sift]
                Y_d.append(rs_1)
                if len(Y_d)==p_count and p_count>0:
                        Y_D.append(Y_d)
                        X_D.append(X_d)
                        S_L.append(seq_id)
                        Y_d=[]
                        X_d=[]
                        F_l=[]
                if len(Y_D)>=max_count and max_count>0:
                    X_D=numpy.asarray(X_D,dtype=numpy.float32)
                    Y_D=numpy.asarray(Y_D,dtype=numpy.float32)
                    return (db,X_D,Y_D,F_L,G_L,S_L)
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
                if len(Y_D)>=max_count and max_count>0:
                    X_D=numpy.asarray(X_D,dtype=numpy.float32)
                    Y_D=numpy.asarray(Y_D,dtype=numpy.float32)
                    return (db,X_D,Y_D,F_L,G_L,S_L)


    X_D=numpy.asarray(X_D,dtype=numpy.float32)
    Y_D=numpy.asarray(Y_D,dtype=numpy.float32)
    return (db,X_D,Y_D,F_L,G_L,S_L)

def joints_sequence_tp12(db,base_file,max_count,p_count,sindex,mode,get_flist=False):
    #LSTM local data
    if mode==0:#load training data.
        lst_act=['S1','S5','S6','S7','S8']
    elif mode==1:#load test data
        lst_act=['S9','S11']
    elif mode==2:#load full data
        lst_act=['S1','S5','S6','S7','S8','S9','S11']
    elif mode==3:#load full data
        lst_act=['S11',]
    elif mode==4:#load full data
        lst_act=['S11']
    else:
        raise Exception('You should pass mode argument for data loading.!') #
    lst_act=['S11']
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
            # if 'Greeting' not in sq:
            #     continue
            X_d=[]
            Y_d=[]
            F_l=[]
            seq_id+=1
            tmp_folder=base_file+actor+"/"+sq+"/"
            if tmp_folder not in db:
                id_list=os.listdir(tmp_folder)
                results=numpy.random.uniform(0.0,1.0,size=(len(id_list),48))
                db[tmp_folder]=results
            else:
                results=db[tmp_folder]

            sift=1
            for r in range(len(results)-(sift+sindex)):
                t_r=r+sindex
                rs=results[t_r]
                X_d.append(rs)
                rs_1=results[t_r+sift]
                Y_d.append(rs_1)
                if len(Y_d)==p_count and p_count>0:
                        Y_D.append(Y_d)
                        X_D.append(X_d)
                        S_L.append(seq_id)
                        Y_d=[]
                        X_d=[]
                        F_l=[]
                if len(Y_D)>=max_count and max_count>-1:
                    X_D=numpy.asarray(X_D,dtype=numpy.float32)
                    Y_D=numpy.asarray(Y_D,dtype=numpy.float32)
                    return (db,X_D,Y_D,F_L,G_L,S_L)
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
                if len(Y_D)>=max_count and max_count>-1:
                    X_D=numpy.asarray(X_D,dtype=numpy.float32)
                    Y_D=numpy.asarray(Y_D,dtype=numpy.float32)
                    return (db,X_D,Y_D,F_L,G_L,S_L)


    X_D=numpy.asarray(X_D,dtype=numpy.float32)
    Y_D=numpy.asarray(Y_D,dtype=numpy.float32)
    return (db,X_D,Y_D,F_L,G_L,S_L)

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

def prepare_lstm_batch(index_list, minibatch_index, batch_size, S_list,LStateList,LStateList_pre, F_list, params, Y, X,state_reset_counter_lst):
    curr_id_lst=index_list[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024
    pre_id_lst=index_list[(minibatch_index-1) * batch_size:(minibatch_index) * batch_size]
    curr_sid=S_list[curr_id_lst]
    pre_sid=S_list[pre_id_lst]
    new_S=[numpy.zeros(shape=(batch_size,params['n_hidden']), dtype=dtype) for i in range(len(LStateList))]
    for idx in range(batch_size):
        state_reset_counter=state_reset_counter_lst[idx]
        if(minibatch_index==0):
            for s in range(len(new_S)):
                new_S[s][idx,:]=LStateList[s][idx,:]
        elif(pre_sid[idx]!=curr_sid[idx]) or ((state_reset_counter%params['reset_state']==0) and state_reset_counter*params['reset_state']>0):
            for s in range(len(new_S)):
                new_S[s][idx,:]=numpy.zeros(shape=(1,params['n_hidden']), dtype=dtype)
            state_reset_counter_lst[idx]=0
        elif (curr_id_lst[idx]==pre_id_lst[idx]): #If value repeated, we should repeat state also
            state_reset_counter_lst[idx]=state_reset_counter-1
            for s in range(len(new_S)):
                new_S[s][idx,:]=LStateList_pre[s][idx,:]
        else:
            for s in range(len(new_S)):
                new_S[s][idx,:]=LStateList[s][idx,:]
    x=X[curr_id_lst]
    y=Y[curr_id_lst]


    return (new_S,x,y,state_reset_counter_lst)

def prepare_cnn_batch(minibatch_index, batch_size, F_list, Y):
    id_lst=range(minibatch_index * batch_size, (minibatch_index + 1) * batch_size,1)
    x_fl=F_list[id_lst].tolist()
    x=multi_thr_load_cnn_batch(my_list=x_fl)
    y=Y[id_lst]
    return (x,y)


def get_seq_indexes(params,S_L):
    bs=params['batch_size']
    new_S_L=[]
    counter=collections.Counter(S_L)
    lst=[list(t) for t  in counter.items()]
    a=numpy.asarray(lst)
    ss=a[a[:,1].argsort()][::-1]
    b_index=0
    new_index_lst=dict()
    b_index_lst=dict()

    for item in ss:
        seq_srt_intex= numpy.sum(a[0:item[0]-1],axis=0)[1]
        seq_end_intex= seq_srt_intex+item[1]
        sub_idx_lst=S_L[seq_srt_intex:seq_end_intex]
        new_S_L.extend(sub_idx_lst)

    for i in range(bs):
        b_index_lst[i]=0
    batch_inner_index=0
    for l_idx in range(len(new_S_L)):
        l=new_S_L[l_idx]
        if(l_idx>0):
            if(l!=new_S_L[l_idx-1]):
                for i in range(bs):
                    if(b_index>b_index_lst[i]):
                        b_index=b_index_lst[i]
                        batch_inner_index=i

        index=b_index*bs+batch_inner_index
        if(index in new_index_lst):
            print 'exist'
        new_index_lst[index]=l_idx
        b_index+=1
        b_index_lst[batch_inner_index]=b_index


    mx=max(b_index_lst.values())
    for b in b_index_lst.keys():
        b_index=b_index_lst[b]
        diff=mx-b_index
        if(diff>0):
            index=(b_index-1)*bs+b
            rpt=new_index_lst[index]
            for inc in range(diff):
                new_index=(b_index+inc)*bs+b
                new_index_lst[new_index]=rpt

    new_lst = collections.OrderedDict(sorted(new_index_lst.items())).values()
    return (new_lst,numpy.asarray(new_S_L))

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
