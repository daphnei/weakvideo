import argparse
import scipy.cluster
import scipy.spatial
import pickle
import os
import numpy as np

import utils

parser = argparse.ArgumentParser()
parser.add_argument('--inputFile', type=str, required=True,
                    help='Location of pickle file outputed ny extract_face_features.py')
parser.add_argument('--numClusters', type=int, required=True,
                    help='The k for k-means')
args = parser.parse_args()

if __name__ == '__main__':
    faces = utils.pickleToFaces(args.inputFile)
    
    allFaceReps = []
    allFaces = []
    for listOfFaces in faces.values():
        allFaces.extend(listOfFaces)
        allFaceReps.extend(face.rep for face in listOfFaces)

    numFaces = len(allFaces)
   
    centroids, labels = scipy.cluster.vq.kmeans2(allFaceReps, k=args.numClusters)
    
    clustersDict = {}
    for k in xrange(0, args.numClusters):
        clustersDict[k] = list(allFaces[j] for j in xrange(0, numFaces) if labels[j]==k)


    # Remove empty clusters
    for key in clustersDict.keys():
        if clustersDict[key] == []:
            clustersDict.pop(key, None)

    faceDim = allFaces[0].image.shape[0]
    utils.visualizeClusters(clustersDict, faceDim)
