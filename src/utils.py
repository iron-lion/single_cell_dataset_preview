import pandas as pd
import numpy as np
import os
import glob
import sys

from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import seaborn as sns


def tsne_get(whole_exp, labels, cluster='kmean'):
    i = 0
    best = (0,0,0)
    tsne_df = TSNE(n_components=2).fit_transform(whole_exp)
    #tsne_df = TSNE(n_components=2, init='pca').fit_transform(whole_exp.cpu().detach().data)
    #tsne_df = whole_exp

    df=pd.DataFrame()
    df['tsne1'] = tsne_df[:,0]
    df['tsne2'] = tsne_df[:,1]
    df['label'] = labels

    if cluster:
        best_kmeans_label = []
        #print(whole_exp, whole_exp.shape, len(whole_key))
        for kn in range(2,len(set(labels))+4):
            if cluster == 'kmean':
                kmeans = KMeans(n_clusters = kn, n_init=20, max_iter=50).fit(tsne_df)
            elif cluster == 'dbscan':
                kmeans = DBSCAN(eps=0.5 * kn, min_samples=10).fit(tsne_df)

            test_ari = adjusted_rand_score(kmeans.labels_, labels)
            if (len(set(kmeans.labels_)) > 1):
                sil = silhouette_score(tsne_df, kmeans.labels_)
            if best[1] < test_ari:
                best = (len(set(kmeans.labels_)), test_ari, sil)
                best_kmeans_label = kmeans.labels_

            #print(Counter(kmeans.labels_), Counter(whole_key))
        print(cluster, '#cluster:', best)
    return best, df


def draw_plot(df_, result_, axs_, label_):
    sns.scatterplot(
        x="tsne1", y="tsne2",
        size=0.3,
        data=df_,
        legend="full",
        hue="label",
        hue_order=sorted(set(label_)),
        alpha=0.8,
        ax=axs_
        )
    axs_.get_legend().remove()
    axs_.spines['top'].set_visible(False)
    axs_.spines['right'].set_visible(False)
    axs_.get_xaxis().set_ticks([])
    axs_.get_yaxis().set_ticks([])
    axs_.set_xlabel('')
    axs_.set_ylabel('')

    txt = "ARI: {ari:.2f}\nSilhouette: {sil:.2f}"
    axs_.text(min(df_['tsne1']),max(df_['tsne2'])*1.2,txt.format(ari = result_[1], sil=result_[2]), fontsize=10, horizontalalignment='left', verticalalignment='center')

