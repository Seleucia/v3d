import numpy
import os
from multiprocessing.dummy import Pool as ThreadPool
from PIL import Image



def load_file_cnn_lstm_patch(fl):
    img = Image.open(fl)
    arr=numpy.asarray(img)
    return arr


base_file="/mnt/hc/joints16/"
istest=0

joints_file=base_file
img_folder=base_file.replace('joints16','h36m_rgb_img_crop')
if istest==0:
    lst_act=['S1','S5','S6','S7','S8']
else:
    lst_act=['S9','S11']
S_L=[]
seq_id=0
for actor in lst_act:
    tmp_folder=joints_file+actor+"/"
    lst_sq=os.listdir(tmp_folder)
    for sq in lst_sq:
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
        results = pool.map(load_file_cnn_lstm_patch, midlayer_list)
        pool.close()
        print numpy.mean(numpy.mean(numpy.mean(results,axis=0),axis=1),axis=1)

