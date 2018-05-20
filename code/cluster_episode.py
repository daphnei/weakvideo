import argparse
import scipy.cluster
import scipy.spatial
from sklearn.cluster import DBSCAN
import pickle
import os
import numpy as np

import utils
import visualization as vis

parser = argparse.ArgumentParser()
parser.add_argument('--inputFile', type=str, required=True,
                    help='Location of pickle file outputed ny extract_face_features.py')
parser.add_argument('--numClusters', type=int, default=-1,
                    help='The k for k-means')
parser.add_argument('--method', type=str, required=True,
                    help='Method for clustering. One of [kmeans, dbscan]')
args = parser.parse_args()

if __name__ == '__main__':
    '''Clusters all of the faces from one episode and visualizes the resulting clusters.'''
    faces = utils.pickleToFaces(args.inputFile)
    
    allFaceReps = []
    allFaces = []
    for listOfFaces in faces.values():
        allFaces.extend(listOfFaces)
        allFaceReps.extend(face.rep for face in listOfFaces)

    import pdb; pdb.set_trace()
    numFaces = len(allFaces)
    print('%d faces detected in episode' %(numFaces))

    if args.method == 'kmeans'
        centroids, labels = scipy.cluster.vq.kmeans2(allFaceReps, k=args.numClusters)
    elif args.method == 'dbscan':
        db = DBSCAN(metric='euclidean', min_samples=4).fit(allFaceReps)
        labels = db.labels_
        numClusters = args.numClusters
        if numClusters == -1:
            numClusters = len(set(x for x in labels if x != -1))
            print('%s out of %d faces have been discarded' % (len(list(x for x in labels if x == -1)), numFaces))

    clustersDict = {}
    for k in xrange(0, numClusters):
        clustersDict[k] = list(allFaces[j] for j in xrange(0, numFaces) if labels[j]==k)

    # Remove empty clusters
    for key in clustersDict.keys():
        if clustersDict[key] == []:
            clustersDict.pop(key, None)

    faceDim = allFaces[0].image.shape[0]
    vis.visualizeClusters(clustersDict, faceDim, outputPath='output.png')
