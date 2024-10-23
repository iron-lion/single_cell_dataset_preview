import pandas as pd
import numpy as np
import os
import glob
import sys

from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import seaborn as sns

import src.utils as my_u
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()

file_list = glob.glob("./dataset/campbell/RAW/GSE93374_Merged_all_020816_DGE.txt")
#file_list = glob.glob("./dataset/campbell/RAW/l")

print(file_list)

LABEL = '7.clust_all'

total_data = pd.DataFrame()
for ff in file_list:
    data = pd.read_csv(ff, sep='\t', index_col=0, header=0)
    total_data = total_data.append(data)
    print(total_data.shape)
    
total_data = total_data.transpose()


labels = pd.read_csv('./dataset/campbell/RAW/GSE93374_cell_metadata.txt', sep='\t', index_col=0, header=0)
labels = labels.filter(total_data.index, axis=0)
labels = labels[LABEL]
print(labels)
#LABEL = labels.colnames

total_data = pd.concat([total_data, labels.transpose()], axis=1)
print(total_data.shape)

labels = total_data[LABEL].values.tolist()
print(set(labels))
total_data.pop(LABEL)
total_data = total_data.astype('float32')
print(total_data.shape)


whole_exp = total_data
raw_result, raw_df = my_u.tsne_get(whole_exp, labels)
#######
whole_exp = total_data
whole_exp = np.log2(whole_exp + 1.0)
log2_result, log2_df = my_u.tsne_get(whole_exp, labels)
#######
whole_exp = total_data
whole_exp = whole_exp.divide(whole_exp.sum(1), axis = 0).mul(20000)         
whole_exp = whole_exp.replace(np.nan,0) 
total_result, total_df = my_u.tsne_get(whole_exp, labels)
#######
whole_exp = total_data
whole_exp = whole_exp.divide(whole_exp.sum(1), axis = 0).mul(20000)         
whole_exp = whole_exp.replace(np.nan,0) 
whole_exp = np.log2(whole_exp + 1.0)
total_log2_result, total_log2_df = my_u.tsne_get(whole_exp, labels)
#######

whole_exp = total_data
np.transpose(min_max_scaler.fit_transform(whole_exp.transpose()))
mm_raw_result, mm_raw_df = my_u.tsne_get(whole_exp, labels)
#######
whole_exp = total_data
whole_exp = np.log2(whole_exp + 1.0)
np.transpose(min_max_scaler.fit_transform(whole_exp.transpose()))
mm_log2_result, mm_log2_df = my_u.tsne_get(whole_exp, labels)
#######
whole_exp = total_data
whole_exp = whole_exp.divide(whole_exp.sum(1), axis = 0).mul(20000)  
whole_exp = whole_exp.replace(np.nan,0) 
np.transpose(min_max_scaler.fit_transform(whole_exp.transpose()))
mm_total_result, mm_total_df = my_u.tsne_get(whole_exp, labels)
#######
whole_exp = total_data
whole_exp = whole_exp.divide(whole_exp.sum(1), axis = 0).mul(20000)         
whole_exp = whole_exp.replace(np.nan,0) 
whole_exp = np.log2(whole_exp + 1.0)
np.transpose(min_max_scaler.fit_transform(whole_exp.transpose()))
mm_total_log2_result, mm_total_log2_df = my_u.tsne_get(whole_exp, labels)



plt.figure(figsize=(16,8), dpi=300)
ax00 = plt.subplot2grid((2,4), (0,0)) 
ax10 = plt.subplot2grid((2,4), (0,1))  
ax20 = plt.subplot2grid((2,4), (0,2))  
ax30 = plt.subplot2grid((2,4), (0,3))  

ax01 = plt.subplot2grid((2,4), (1,0)) 
ax11 = plt.subplot2grid((2,4), (1,1))  
ax21 = plt.subplot2grid((2,4), (1,2))  
ax31 = plt.subplot2grid((2,4), (1,3))  

my_u.draw_plot(raw_df, raw_result, ax00, labels)
my_u.draw_plot(log2_df, log2_result, ax10, labels)
my_u.draw_plot(total_df, total_result, ax20, labels)
my_u.draw_plot(total_log2_df, total_log2_result, ax30, labels)
my_u.draw_plot(mm_raw_df, mm_raw_result, ax01, labels)
my_u.draw_plot(mm_log2_df, mm_log2_result, ax11, labels)
my_u.draw_plot(mm_total_df, mm_total_result, ax21, labels)
my_u.draw_plot(mm_total_log2_df, mm_total_log2_result, ax31, labels)

ax00.set_ylabel('raw' , fontsize=14)

ax01.set_xlabel('raw', fontsize=13)
ax11.set_xlabel('log2', fontsize=13)
ax21.set_xlabel('total', fontsize=13)
ax31.set_xlabel('total_log2', fontsize=13)
ax01.set_ylabel('min-max normalized' , fontsize=14)
ax31.legend(bbox_to_anchor=(1.1,0), loc='lower left',borderaxespad=0)

plt.save('tet.png')
