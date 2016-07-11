import numpy
import os
import collections
from collections import OrderedDict
import collections
from multiprocessing.dummy import Pool as ThreadPool
from PIL import Image


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
            id_list=os.listdir(tmp_folder)
            joint_list=[tmp_folder + p1 for p1 in id_list]
            # pool = ThreadPool(300)
            # results = pool.map(load_file, joint_list)
            # pool.close()
            results=numpy.random.uniform(0.0,1.0,size=(len(id_list),3))
            sift=1
            print "s: %s, cnt:%d"%(sq.split('.')[0],int(len(results)))
            for r in range(len(results)-sift):
                rs=results[r]
                X_d.append(rs)
                rs_1=results[r+sift]
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
                if len(Y_D)>=max_count and max_count>-1:
                    X_D=numpy.asarray(X_D,dtype=numpy.float32)
                    Y_D=numpy.asarray(Y_D,dtype=numpy.float32)
                    return (X_D,Y_D,F_L,G_L,S_L)


    X_D=numpy.asarray(X_D,dtype=numpy.float32)
    Y_D=numpy.asarray(Y_D,dtype=numpy.float32)
    return (X_D,Y_D,F_L,G_L,S_L)

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

base_file="/mnt/Data1/hc/joints16/"
base_file="/home/coskun/PycharmProjects/data/pose/joints16/"
params=dict()
params['batch_size']=10
istest=0
total=0
s_total=0

(X_D,Y_D,F_L,G_L,S_L)=joints_sequence_tp1(base_file,max_count=-1,p_count=50,sindex=0,mode=2)
(index_train_list,S_Train_list)=get_seq_indexes(params,S_L)

print S_Train_list[index_train_list]
print 'Completed'