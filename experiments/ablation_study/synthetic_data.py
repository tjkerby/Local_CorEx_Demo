import matplotlib.pyplot as plt
import numpy as np
from hashlib import sha1
from sklearn.metrics import precision_recall_curve as prc 
from sklearn.metrics import auc
import phate

# import sys
# sys.path.append('/home/TomKerby/Research/local_corex_private')
from local_corex import BioCorex, LinearCorex

from scipy.spatial.distance import cosine
# from scipy.stats import wasserstein_distance as emd

from sklearn.cluster import KMeans
from collections import Counter

import pandas as pd
from time import time

def create_cov_mat_clust_1(visualize=False):
    mat = np.diag(np.hstack([np.ones(5)*7,np.ones(5)*3, np.ones(5)*10, np.ones(5)*5, np.ones(5)*8])).reshape(25,25)
    first_block = [(x,y) for x in range(5) for y in range(5) if x != y]
    second_block = [(12, x) for x in range(5,10)]
    second_block.extend([(x,y) for x in range(5,10) for y in range(5,10) if x != y])
    third_block = [(x,y) for x in range(10,15) if x != 12 for y in range(10,15) if y != 12 if y != x]
    fourth_block = [(x,y) for x in range(15,21) for y in range(15,21) if x != y]
    fifth_block = [(x,y) for x in range(21,25) for y in range(21,25) if x != y]
    for edge in first_block:
        mat[edge] = 5
    for edge in second_block:
        mat[edge] = 2
        mat[edge[::-1]] = 2
    for edge in third_block:
        mat[edge] = 2
    for edge in fourth_block:
        mat[edge] = 4
        mat[edge[::-1]] = 4
    for edge in fifth_block:
        mat[edge] = 3
    if visualize:
        plt.imshow(mat)
    return mat

def create_cov_mat_clust_2(visualize=False):
    mat = np.diag(np.hstack([np.ones(5)*7,np.ones(5)*3, np.ones(5)*10, np.ones(5)*5, np.ones(5)*8])).reshape(25,25)
    first_block = [(x,y) for x in range(5) for y in range(5) if x != y]
    second_block = [(12, x) for x in range(5,10)]
    second_block.extend([(x,y) for x in range(5,10) for y in range(5,10) if x != y])
    second_block.extend([(x,y) for x in range(15,20) for y in range(5,10) if x != y])
    third_block = [(x,y) for x in range(10,15) if x != 12 for y in range(10,15) if y != 12 if y != x]
    fourth_block = [(12, x) for x in range(15,20)]
    fourth_block.extend([(x,y) for x in range(15,20) for y in range(15,20) if x != y])
    # fourth_block.extend([(x,y) for x in range(5,10) for y in range(15,20) if x != y])
    fifth_block = [(x,y) for x in range(20,25) for y in range(20,25) if x != y]
    for edge in first_block:
        mat[edge] = 5
    for edge in second_block:
        mat[edge] = 2
        mat[edge[::-1]] = 2
    for edge in third_block:
        mat[edge] = 2
    for edge in fourth_block:
        mat[edge] = 4
        mat[edge[::-1]] = 4
    for edge in fifth_block:
        mat[edge] = 3
    if visualize:
        plt.imshow(mat)
    return mat

def create_cov_mat_clust_3(visualize=False):
    mat = np.diag(np.hstack([np.ones(5)*7,np.ones(5)*3, np.ones(5)*10, np.ones(5)*5, np.ones(5)*8])).reshape(25,25)
    first_block = [(x,y) for x in range(5) for y in range(5) if x != y]
    second_block = [(12, x) for x in range(5,10)]
    second_block.extend([(x,y) for x in range(5,10) for y in range(5,10) if x != y])
    third_block = [(x,y) for x in range(10,15) if x != 12 for y in range(10,15) if y != 12 if y != x]
    fourth_block = [(12, x) for x in range(15,20)]
    fourth_block.extend([(x,y) for x in range(15,20) for y in range(15,20) if x != y])
    fifth_block = [(x,y) for x in range(20,25) for y in range(20,25) if x != y]
    for edge in first_block:
        mat[edge] = 5
    for edge in second_block:
        mat[edge] = 2
        mat[edge[::-1]] = 2
    for edge in third_block:
        mat[edge] = 2
    for edge in fourth_block:
        mat[edge] = 4
        mat[edge[::-1]] = 4
    for edge in fifth_block:
        mat[edge] = 3
    if visualize:
        plt.imshow(mat)
    return mat

def create_cov_mat_clust_4(visualize=False):
    mat = np.diag(np.hstack([np.ones(5)*7,np.ones(5)*6, np.ones(5)*10, np.ones(5)*6, np.ones(5)*8])).reshape(25,25)
    first_block = [(5*x,5*y) for x in range(5) for y in range(5) if x != y]
    second_block = [(5*x+1,5*y+1) for x in range(5) for y in range(5) if x != y]
    third_block = [(5*x+2,5*y+2) for x in range(5) for y in range(5) if x != y]
    fourth_block = [(5*x+3,5*y+3) for x in range(5) for y in range(5) if x != y]
    fifth_block = [(5*x+4,5*y+4) for x in range(5) for y in range(5) if x != y]
    for edge in first_block:
        mat[edge] = 5
    for edge in second_block:
        mat[edge] = 2
        mat[edge[::-1]] = 2
    for edge in third_block:
        mat[edge] = 2
    for edge in fourth_block:
        mat[edge] = 4
        mat[edge[::-1]] = 4
    for edge in fifth_block:
        mat[edge] = 3
    if visualize:
        plt.imshow(mat)
    return mat

def gen_clustered_sample(same_dist=True, alpha=0, cluster_size=100, disjoint=False):
    clust_means_1 = np.hstack([np.ones(5)*5, np.ones(5)*10, np.ones(5), np.ones(5)*10, np.ones(5)*3])
    clust_means_2 = np.hstack([np.ones(5)*2.5, np.ones(5)*6, np.ones(5)*3, np.ones(5), np.ones(5)*10])
    diff_means = clust_means_1 - clust_means_2
    mat = create_cov_mat_clust_1(visualize=False)
    clust_1 = np.random.multivariate_normal(clust_means_1, mat, size=cluster_size)
    if same_dist:
        mat_2 = mat
    else:
        if disjoint:
            mat_2 = create_cov_mat_clust_4(visualize=False)
        else:
            mat_2 = create_cov_mat_clust_2(visualize=False)       
    clust_2 = np.random.multivariate_normal(clust_means_1 + alpha*diff_means, mat_2, size=cluster_size)
    return clust_1, clust_2

def generate_and_partition_data(alpha, cluster_size, disjoint=False):
    clust_1, clust_2 = gen_clustered_sample(same_dist=False, alpha=alpha, cluster_size=cluster_size, disjoint=disjoint)
    full_clust = np.concatenate([clust_1, clust_2], axis=0)
    labels = np.concatenate([np.repeat('clust_1', cluster_size), np.repeat('clust_2', cluster_size)])
    st = time()
    phate_operator = phate.PHATE(n_components=5, n_jobs=4, random_state=42)
    Y_phate = phate_operator.fit_transform(full_clust)
    num_clusters = 2
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init = 'auto') 
    kmeans.fit(Y_phate)
    pred = kmeans.fit_predict(Y_phate)
    indexes = []
    for i in range(0, num_clusters):
        index = pred == i
        indexes.append(index)
    et = time()
    return full_clust, labels, indexes, et - st

def run_simulation(rep, 
                   cluster_size=100, 
                   alphas=[x/10 for x in range(11)], 
                   num_factors=[x for x in range(3,8)], 
                   true_groups=5, 
                   bio=True, 
                   disjoint=False,
                   visualize_vars=False,
                   verbose=False):
    cols = ['num_factors', 'alpha', 'rep_num', 'class_prop', 'method', 'groundtruth', 
            'ave_group_cosine', 'ave_auc_group_prc', 'ave_lf_cosine', 'ave_auc_lf_prc', 'total_time', 'corex_time']
    cols.extend([f'ave_auc_lf_{i}_prc' for i in range(true_groups)])
    cols.extend([f'ave_lf_{i}_cosine' for i in range(true_groups)])
    cols.extend(['top_k_ave_auc_lf_prc', 'top_k_ave_lf_cosine'])
    full_values = []
    rounds_per_rep = len(num_factors)*len(alphas)*2
    tick = rounds_per_rep*rep
    for j, alpha in enumerate(alphas):
        full_clust, labels, indexes, partition_time = generate_and_partition_data(alpha, cluster_size, disjoint=disjoint)
        for k, num_factor in enumerate(num_factors):
            for l, group in enumerate([True, False]):
                tick += 1
                print(f'\n\tStarted {tick} out of {rounds_per_rep*16} rounds...\n')
                if group:            
                    groups = get_groups_from_mat(create_cov_mat_clust_1(visualize=False))
                    clust = 'clust_1'
                    ind = 0
                else:
                    if disjoint:
                        groups = get_groups_from_mat(create_cov_mat_clust_4(visualize=False))
                    else:
                        groups = get_groups_from_mat(create_cov_mat_clust_2(visualize=False))
                    clust = 'clust_2'
                    ind = 1
                
                ### Linear Corex ###
                if visualize_vars:
                    print(f"Linear Corex with {num_factor} factors on alpha level {alpha}")
                st = time()
                lin_cor_model = LinearCorex(n_hidden=num_factor, seed=42, gaussianize='outliers')
                __ = lin_cor_model.fit_transform(full_clust)
                et = time()
                group_cosine_score = score_group_coverage(groups, lin_cor_model, plot=visualize_vars, dist=cosine, verbose=verbose)
                group_auc_prc_score = score_group_coverage(groups, lin_cor_model, plot=visualize_vars, dist="auc_prc", verbose=verbose)
                lf_cosine_scores = score_lf_quality(groups, lin_cor_model, plot=visualize_vars, dist=cosine, verbose=verbose)
                lf_auc_prc_scores = score_lf_quality(groups, lin_cor_model, plot=visualize_vars, dist="auc_prc", verbose=verbose)
                values = [num_factor, alpha, rep, .5, 'linear_corex', clust, group_cosine_score, 
                    group_auc_prc_score, np.mean(lf_cosine_scores), np.mean(lf_auc_prc_scores), et - st, et - st]
                values.extend([lf_auc_prc_scores[i] if i < num_factor else np.nan for i in range(true_groups)])
                values.extend([lf_cosine_scores[i] if i < num_factor else np.nan for i in range(true_groups)])
                values.extend([np.mean(lf_auc_prc_scores) if num_factor <= true_groups else np.mean(lf_auc_prc_scores[:true_groups]),
                               np.mean(lf_cosine_scores) if num_factor <= true_groups else np.mean(lf_cosine_scores[:true_groups])])
                full_values.append(pd.DataFrame([values], columns=cols))
                
                if bio:
                    ### Bio Corex ###
                    if visualize_vars:
                        print(f"Bio Corex with {num_factor} factors on alpha level {alpha}")
                    st = time()
                    layer1 = BioCorex(n_hidden=num_factor, dim_hidden=2, marginal_description='gaussian', smooth_marginals=False) 
                    __ = layer1.fit_transform(full_clust)
                    et = time()
                    group_cosine_score = score_group_coverage(groups, layer1, False, plot=visualize_vars, dist=cosine, verbose=verbose)
                    group_auc_prc_score = score_group_coverage(groups, layer1, False, plot=visualize_vars, dist="auc_prc", verbose=verbose)
                    lf_cosine_scores = score_lf_quality(groups, layer1, False, plot=visualize_vars, dist=cosine, verbose=verbose)
                    lf_auc_prc_scores = score_lf_quality(groups, layer1, False, plot=visualize_vars, dist="auc_prc", verbose=verbose)
                    values = [num_factor, alpha, rep, .5, 'bio_corex', clust, group_cosine_score, 
                        group_auc_prc_score, np.mean(lf_cosine_scores), np.mean(lf_auc_prc_scores), et - st, et - st]
                    values.extend([lf_auc_prc_scores[i] if i < num_factor else np.nan for i in range(true_groups)])
                    values.extend([lf_cosine_scores[i] if i < num_factor else np.nan for i in range(true_groups)])
                    values.extend([np.mean(lf_auc_prc_scores) if num_factor <= true_groups else np.mean(lf_auc_prc_scores[:true_groups]),
                                np.mean(lf_cosine_scores) if num_factor <= true_groups else np.mean(lf_cosine_scores[:true_groups])])
                    full_values.append(pd.DataFrame([values], columns=cols))
                
                clust_data = full_clust[indexes[ind]]
                ratio_correct = Counter(labels[indexes[ind]])[clust] / (Counter(labels[indexes[ind]])['clust_1'] + Counter(labels[indexes[ind]])['clust_2'])

                ### Local Linear Corex ###
                if visualize_vars:
                    print(f"Local Linear Corex with {num_factor} factors on alpha level {alpha}")
                st = time()
                lin_cor_model = LinearCorex(n_hidden=num_factor, seed=42, gaussianize='outliers') #, discourage_overlap=False)
                __ = lin_cor_model.fit_transform(clust_data)
                et = time()
                group_cosine_score = score_group_coverage(groups, lin_cor_model, plot=visualize_vars, dist=cosine, verbose=verbose)
                group_auc_prc_score = score_group_coverage(groups, lin_cor_model, plot=visualize_vars, dist="auc_prc", verbose=verbose)
                lf_cosine_scores = score_lf_quality(groups, lin_cor_model, plot=visualize_vars, dist=cosine, verbose=verbose)
                lf_auc_prc_scores = score_lf_quality(groups, lin_cor_model, plot=visualize_vars, dist="auc_prc", verbose=verbose)
                values = [num_factor, alpha, rep, ratio_correct, 'loc_lin_corex', clust, group_cosine_score, 
                    group_auc_prc_score, np.mean(lf_cosine_scores), np.mean(lf_auc_prc_scores), et - st + partition_time, et - st]
                values.extend([lf_auc_prc_scores[i] if i < num_factor else np.nan for i in range(true_groups)])
                values.extend([lf_cosine_scores[i] if i < num_factor else np.nan for i in range(true_groups)])
                values.extend([np.mean(lf_auc_prc_scores) if num_factor <= true_groups else np.mean(lf_auc_prc_scores[:true_groups]),
                               np.mean(lf_cosine_scores) if num_factor <= true_groups else np.mean(lf_cosine_scores[:true_groups])])
                full_values.append(pd.DataFrame([values], columns=cols))
                
                if bio:
                    ### Local Bio Corex ###
                    if visualize_vars:
                        print(f"Local Bio Corex with {num_factor} factors on alpha level {alpha}")
                    st = time()
                    layer1 = BioCorex(n_hidden=num_factor, dim_hidden=2, marginal_description='gaussian', smooth_marginals=False) 
                    __ = layer1.fit_transform(clust_data)
                    et = time()
                    group_cosine_score = score_group_coverage(groups, layer1, False, plot=visualize_vars, dist=cosine, verbose=verbose)
                    group_auc_prc_score = score_group_coverage(groups, layer1, False, plot=visualize_vars, dist="auc_prc", verbose=verbose)
                    lf_cosine_scores = score_lf_quality(groups, layer1, False, plot=visualize_vars, dist=cosine, verbose=verbose)
                    lf_auc_prc_scores = score_lf_quality(groups, layer1, False, plot=visualize_vars, dist="auc_prc", verbose=verbose)
                    values = [num_factor, alpha, rep, ratio_correct, 'loc_bio_corex', clust, group_cosine_score, 
                        group_auc_prc_score, np.mean(lf_cosine_scores), np.mean(lf_auc_prc_scores), et - st + partition_time, et - st]
                    values.extend([lf_auc_prc_scores[i] if i < num_factor else np.nan for i in range(true_groups)])
                    values.extend([lf_cosine_scores[i] if i < num_factor else np.nan for i in range(true_groups)])
                    values.extend([np.mean(lf_auc_prc_scores) if num_factor <= true_groups else np.mean(lf_auc_prc_scores[:true_groups]),
                                np.mean(lf_cosine_scores) if num_factor <= true_groups else np.mean(lf_cosine_scores[:true_groups])])
                    full_values.append(pd.DataFrame([values], columns=cols))
    return full_values

class HashableNdarray(np.ndarray):
    @classmethod
    def create(cls, array):
        return HashableNdarray(shape=array.shape, dtype=array.dtype, buffer=array.copy())

    def __hash__(self):
        if not hasattr(self, '_HashableNdarray__hash'):
            self.__hash = int(sha1(self.view()).hexdigest(), 16)
        return self.__hash

    def __eq__(self, other):
        if not isinstance(other, HashableNdarray):
            return super().__eq__(other)
        return super().__eq__(super(HashableNdarray, other)).all()
    
def get_groups_from_mat(mat):
    groups = set()
    for i in mat:
        groups.add((i != 0).astype(float).view(HashableNdarray))
    return [np.array(x) for x in groups]

def auc_prc(group, lf):
    prec, rec, thresh = prc(group, lf)
    return auc(rec, prec)

def score_group_coverage(groups, corex_model, linear=True, verbose=False, plot=False, dist=cosine):  
    if plot:
        print("\tBEST LF FOR EACH GROUP")
        fig, axes = plt.subplots(len(groups), 1, figsize=(8,1*len(groups)))
    if dist == "auc_prc":
        dist = auc_prc
    scores = []
    if linear:
        for j, group in enumerate(groups):
            ind = 0
            if dist == auc_prc:
                min_score = 0
            else:
                min_score = np.inf
            if verbose:
                print(min_score)
            for i in range(len(corex_model.moments["MI"])):
                score = dist(group, corex_model.moments["MI"][i])
                if dist == auc_prc:
                    if score > min_score:
                        min_score = score
                        ind = i
                else:
                    if score < min_score:
                        min_score = score
                        ind = i
                if verbose:
                    print(min_score)
            scores.append(min_score)
            if plot:
                axes[j].imshow(np.concatenate([group / np.sum(group), corex_model.moments["MI"][ind] / np.sum(corex_model.moments["MI"][ind])], axis=0).reshape(2,25))
    else:
        for j, group in enumerate(groups):
            ind = 0
            if dist == auc_prc:
                min_score = 0
            else:
                min_score = np.inf
            for i in range(len(corex_model.mis)):
                score = dist(group, corex_model.mis[i])
                if dist == auc_prc:
                    if score > min_score:
                        min_score = score
                        ind = i
                else:
                    if score < min_score:
                        min_score = score
                        ind = i
                if verbose:
                    print(min_score)
            scores.append(min_score)
            if plot:
                axes[j].imshow(np.concatenate([group / np.sum(group), corex_model.mis[ind] / np.sum(corex_model.mis[ind])], axis=0).reshape(2,25))
    plt.show()
    return(np.mean(scores))

def score_lf_quality(groups, corex_model, linear=True, verbose=False, plot=False, dist=cosine):
    if plot:
        print("\tBEST GROUP FOR EACH LF")
        if linear:
            fig, axes = plt.subplots(len(corex_model.moments["MI"]), 1, figsize=(8,1*len(corex_model.moments["MI"])))
        else:
            fig, axes = plt.subplots(len(corex_model.mis), 1, figsize=(8,1*len(corex_model.mis)))
    if dist == "auc_prc":
        dist = auc_prc
    scores = []
    if linear:
        for j, lf in enumerate(corex_model.moments["MI"]):
            ind = 0
            if dist == auc_prc:
                min_score = 0
            else:
                min_score = np.inf
            for i in range(len(groups)):
                score = dist(groups[i], lf)
                if dist == auc_prc:
                    if score > min_score:
                        min_score = score
                        ind = i
                else:
                    if score < min_score:
                        min_score = score
                        ind = i
            if verbose:
                print(min_score)
            scores.append(min_score)
            if plot:
                axes[j].imshow(np.concatenate([groups[ind] / np.sum(groups[ind]), lf / np.sum(lf)], axis=0).reshape(2,25))
    else:
        for j, lf in enumerate(corex_model.mis):
            ind = 0
            if dist == auc_prc:
                min_score = 0
            else:
                min_score = np.inf
            for i in range(len(groups)):
                score = dist(groups[i], lf)
                if dist == auc_prc:
                    if score > min_score:
                        min_score = score
                        ind = i
                else:
                    if score < min_score:
                        min_score = score
                        ind = i
            if verbose:
                print(min_score)
            scores.append(min_score)
            if plot:
                axes[j].imshow(np.concatenate([groups[ind] / np.sum(groups[ind]), lf / np.sum(lf)], axis=0).reshape(2,25))
    plt.show()
    return(scores)