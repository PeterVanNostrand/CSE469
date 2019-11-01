
# coding: utf-8

import sys
import os
from matplotlib import pyplot as plt
import numpy as np
import copy
import csv

rowToClust = []

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split(',')
        fltLine = list(map(float,curLine)) #map all elements to float()
        dataMat.append(fltLine)
    return np.array(dataMat)


def merge_cluster(distance_matrix, cluster_candidate, T):
    ''' Merge two closest clusters according to min distances
    1. Find the smallest entry in the distance matrix—suppose the entry 
        is i-th row and j-th column
    2. Merge the clusters that correspond to the i-th row and j-th column 
        of the distance matrix as a new cluster with index T

    Parameters:
    ------------
    distance_matrix : 2-D array
        distance matrix
    cluster_candidate : dictionary
        key is the cluster id, value is point ids in the cluster
    T: int
        current cluster index

    Returns:
    ------------
    cluster_candidate: dictionary
        updated cluster dictionary after merging two clusters
        key is the cluster id, value is point ids in the cluster
    merge_list : list of tuples
        records the two old clusters' id and points that have just been merged.
        [(cluster_one_id, point_ids_in_cluster_one), 
         (cluster_two_id, point_ids_in_cluster_two)]
    '''
    merge_list = []
    minDist = np.min(distance_matrix, axis=None) # find the minimum distance in the array
    minIndex = np.where(distance_matrix == minDist)[0] # find the fist occurance of that min value

    # Indices of minimum distance
    i = minIndex[0] # i-th row
    j = minIndex[1] # j-th column

    # The cluster IDs corresponding to the i,j row/cols
    clustID1 = rowToClust[i]
    clustID2 = rowToClust[j]

    # Get the points from each cluster
    points1 = cluster_candidate[clustID1]
    points2 = cluster_candidate[clustID2]
    newPoints = points1 + points2

    # Remove the old clusters and add a new merged cluser
    del cluster_candidate[clustID1]
    del cluster_candidate[clustID2]
    cluster_candidate[T] = newPoints

    # Record which clusters were merged
    merge_list = [(clustID1, points1), (clustID2, points2)]

    return cluster_candidate, merge_list


def update_distance(distance_matrix, cluster_candidate, merge_list, T):
    ''' Update the distantce matrix
    
    Parameters:
    ------------
    distance_matrix : 2-D array
        distance matrix
    cluster_candidate : dictionary
        key is the updated cluster id, value is point ids in the cluster
    merge_list : list of tuples
        records the two old clusters' id and points that have just been merged.
        [(cluster_one_id, point_ids_in_cluster_one), 
         (cluster_two_id, point_ids_in_cluster_two)]

    Returns:
    ------------
    distance_matrix: 2-D array
        updated distance matrix       
    '''

    # Get which clusters were merged
    clustID1 = merge_list[0][0]
    clustID2 = merge_list[1][0]

    # Get the corresponding row/col values
    global rowToClust
    i = min(rowToClust.index(clustID1), rowToClust.index(clustID2))
    j = max(rowToClust.index(clustID1), rowToClust.index(clustID2))

    # Calculate the new distance between each cluster and the merged cluster
    newDists = {}
    for row in range(0, distance_matrix.shape[0]):
        if (row==i or row==j):
            continue
        newDists[rowToClust[row]] = min(distance_matrix[row][i], distance_matrix[row][j])

    # Remove j-th row and update rowToClust mapping
    newDistMat = np.delete(distance_matrix, j, axis=0)
    for idx in range(j, len(rowToClust)-1):
        rowToClust[idx] = rowToClust[idx+1] 

    # Remove i-th row and update rowToClust mapping
    newDistMat = np.delete(newDistMat, i, axis=0)
    for idx in range(i, len(rowToClust)-1):
        rowToClust[idx] = rowToClust[idx+1]
    
    # Remove i-th and j-th cols, mapping already updated
    newDistMat = np.delete(newDistMat, j, axis=1)
    newDistMat = np.delete(newDistMat, i, axis=1)

    # Add a new row to the bottom and update rowToClust mapping
    newRow = [0] * newDistMat.shape[1]
    newDistMat = np.vstack((newDistMat, newRow))
    lastRow = newDistMat.shape[0] - 1

    # Fill the row with the new distances
    rowToClust[lastRow] = T
    for col in range(0, newDistMat.shape[1]):
        newDistMat[lastRow][col] = newDists[rowToClust[col]]

    # Add new column to right, as matrix is symmetric just use the transpose of new row
    newCol = np.append(newDistMat[lastRow], 100000) # Adding the self-self dist in bottom right corner
    newDistMat = np.vstack((newDistMat.T, newCol)).T
    
    distance_matrix = newDistMat
    return distance_matrix

    

def agglomerative_with_min(data, cluster_number):
    """
    agglomerative clustering algorithm with min link

    Parameters:
    ------------
    data : 2-D array
        each row represents an observation and 
        each column represents an attribute

    cluster_number : int
        number of clusters

    Returns:
    ------------
    clusterAssment: list
        assigned cluster id for each data point
    """
    cluster_candidate = {}
    N = len(data)
    # initialize cluster, each sample is a single cluster at the beginning
    for i in range(N):
        cluster_candidate[i+1] = [i]  #key: cluser id; value: point ids in the cluster

    # initialize distance matrix
    distance_matrix = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if j == i: # or j<=i
                distance_matrix[i,j] = 100000
            else:
                distance_matrix[i,j] = np.sqrt(np.sum((data[i]-data[j])**2))

    global rowToClust
    for i in range(1, N+1):
        rowToClust.append(i)
    
    # hiearchical clustering loop
    T = N + 1 #cluster index
    for i in range(N-cluster_number):
        cluster_candidate, merge_list = merge_cluster(distance_matrix, cluster_candidate, T)
        distance_matrix   = update_distance(distance_matrix, cluster_candidate, merge_list, T)
        print('%d-th merging: %d, %d, %d'% (i, merge_list[0][0], merge_list[1][0], T))
        T += 1
        # print(cluster_candidate)


    # assign new cluster id to each data point 
    clusterAssment = [-1] * N
    for cluster_index, cluster in enumerate(cluster_candidate.values()):
        for c in cluster:
            clusterAssment[c] = cluster_index
    # print (clusterAssment)
    return clusterAssment


def saveData(save_filename, data, clusterAssment):
    clusterAssment = np.array(clusterAssment, dtype = object)[:,None]
    data_cluster = np.concatenate((data, clusterAssment), 1)
    data_cluster = data_cluster.tolist()

    os.makedirs(os.path.dirname(save_filename), exist_ok=True)
    with open(save_filename, 'w', newline = '') as f:
        writer = csv.writer(f)
        writer.writerows(data_cluster)
    f.close()



if __name__ == '__main__':
    if len(sys.argv) == 3:
        data_filename = sys.argv[1]
        cluster_number = int(sys.argv[2])
    else:
        data_filename = 'Example.csv'
        cluster_number = 1

    save_filename = data_filename.replace('.csv', '_hc_cluster.csv')
    save_filename = save_filename.replace('datasets', 'output')

    data = loadDataSet(data_filename)

    clusterAssment = agglomerative_with_min(data, cluster_number)

    saveData(save_filename, data, clusterAssment)
