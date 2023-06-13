import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr

def get_ground_truth_vec(gt_length, loc) -> np.array:
    '''
    Create a [0,1] vector of the size of the relevance (CSL+Seq+SEP) with the motif locations as 1's and the rest as 0's

    Parameter:
        loc : protein.location i.e. [[10,12]]
        prim: protein.primary  i.e. "ABCDEFGHIJKL"
    Output:
        ground_truth_array i.e. array([0,0,0,0,0,0,0,0,0,0,1,1,1,0]) accounting for CLS,Sep token
    '''
    loc_dims = len(loc)
    arr = np.zeros(gt_length)  # 0's array of size protein sequence
    for dim in range(loc_dims):

        # original locations array count from 1 therefore start-1
        start, stop = loc[dim][0], loc[dim][-1]
        start -= 1  # stop value can stay the same because of slicing exclusion
        arr[start:stop] = 1

    motif_location_arr = np.append(arr, 0)
    # adding zero to account for CLS,SEP token
    motif_location_arr = np.append(0, motif_location_arr)

    return motif_location_arr


def pbr(relevance_vector, locations) -> float:
    '''
    Point biserial correlation a.k.a. Pearson correlation, using https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pointbiserialr.html
    '''
    motif_location_arr = get_ground_truth_vec(len(relevance_vector)-2, locations)
    res = pointbiserialr(motif_location_arr, relevance_vector)

    return res.correlation # correlation coefficient r
    

def rra(relevance_vector, locations) -> float:
    '''
    Relevance Rank Accuracy
        K is the number of Amino acids in the motifs of the protein.

    Parameter:
        relevance_vector: vector with attribution values
        locations: [[1,2]] locations of the motifs

    Return:
        Rank Relevance Accuracy Score
    '''
    K = 0
    for i in range(len(locations)):
        locs = locations[i]
        start = locs[0]
        if len(locs)==2: #case [1,4] we want the values of the slice arr[1:5] because the end is exclusive
            end = locs[1]+1
        elif len(locs)==1: #case [23] even if its the last element in the list we can slice a +1
            end = locs[0]+1
        K += end-start

    motif_location_arr = get_ground_truth_vec(len(relevance_vector)-2, locations)
    # returns the position of the highes k relevances
    highest_relevances = relevance_vector.argsort()[::-1][:K]
    p_top_k = np.zeros_like(relevance_vector)  # create 0 array
    p_top_k[highest_relevances] = 1
    
    a = np.dot(p_top_k, motif_location_arr)
    rank_accuracy = a/K
    
    return rank_accuracy

