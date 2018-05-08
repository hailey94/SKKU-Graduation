import csv
from math import floor
from random import shuffle
import os
import shutil
import errno

val_portion=0.1
cluster_0=[]
cluster_1=[]
cluster_2=[]


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.exists(d):
            try:
                shutil.rmtree(d)
            except Exception as e:
                print (e)
                os.unlink(d)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

with open('split_data.csv', newline='') as csvfile:
    data_reader = csv.DictReader(csvfile)
    for row in data_reader:
        #print(row['image_id'], row['hsv_cluster'])
        if (row['hsv_cluster']=='0'):
            cluster_0.append(row['image_id'])
        elif (row['hsv_cluster']=='1'):
            cluster_1.append(row['image_id'])
        elif (row['hsv_cluster']=='2'):
            cluster_2.append(row['image_id'])
        else:
            print ('More cluster')

    # print ('=================')
    # print (cluster_0)
    # print ('=================')
    # print (cluster_1)
    # print ('=================')
    # print (cluster_2)


num_val_0=floor(len(cluster_0)*val_portion)
num_val_1=floor(len(cluster_1)*val_portion)
num_val_2=floor(len(cluster_2)*val_portion)
print ('Number of validation ')
print (num_val_0,num_val_1,num_val_2)
shuffle(cluster_0)
shuffle(cluster_1)
shuffle(cluster_2)

val_ids_0=cluster_0[:num_val_0]
train_ids_0=cluster_0[num_val_0:]

val_ids_1 = cluster_1[:num_val_1]
train_ids_1 = cluster_1[num_val_1:]

val_ids_2 = cluster_2[:num_val_2]
train_ids_2 = cluster_2[num_val_2:]
print ('Number of Training ')
print (len(cluster_0)-num_val_0,len(cluster_1)-num_val_1,len(cluster_0)-num_val_0)

ROOT_DATASET='dataset'
ORIGINAL_DATASET='./kaggle-dsbowl-2018-dataset-fixes/stage1_train'
TRAIN_FOLDERS=['stage_1_c0_train','stage_1_c1_train','stage_1_c2_train']
VAL_FOLDERS=['stage_1_c0_validation','stage_1_c1_validation','stage_1_c2_validation']
train_full_path=os.path.join(ORIGINAL_DATASET)

for i in range (len(TRAIN_FOLDERS)):
    train_path=os.path.join(ROOT_DATASET,TRAIN_FOLDERS[i])
    if not os.path.exists(train_path):
        os.makedirs(train_path)

    if (i==0):
        for j in range(len(train_ids_0)):
            src_file = os.path.join(train_full_path, train_ids_0[j])
            des_file = os.path.join(train_path, train_ids_0[j])
            copytree(src_file, des_file)
    elif (i==1):
        for j in range(len(train_ids_1)):
            src_file = os.path.join(train_full_path, train_ids_1[j])
            des_file = os.path.join(train_path, train_ids_1[j])
            copytree(src_file, des_file)
    elif (i==2):
        for j in range(len(train_ids_2)):
            src_file = os.path.join(train_full_path, train_ids_2[j])
            des_file = os.path.join(train_path, train_ids_2[j])
            copytree(src_file, des_file)


#Valdidation

for i in range (len(VAL_FOLDERS)):
    val_path=os.path.join(ROOT_DATASET,VAL_FOLDERS[i])
    if not os.path.exists(val_path):
        os.makedirs(val_path)

    if (i==0):
        for j in range(len(val_ids_0)):
            src_file = os.path.join(train_full_path, val_ids_0[j])
            des_file = os.path.join(val_path, val_ids_0[j])
            copytree(src_file, des_file)
    elif (i==1):
        for j in range(len(val_ids_1)):
            src_file = os.path.join(train_full_path, val_ids_1[j])
            des_file = os.path.join(val_path, val_ids_1[j])
            copytree(src_file, des_file)
    elif (i==2):
        for j in range(len(val_ids_2)):
            src_file = os.path.join(train_full_path, val_ids_2[j])
            des_file = os.path.join(val_path, val_ids_2[j])
            copytree(src_file, des_file)
