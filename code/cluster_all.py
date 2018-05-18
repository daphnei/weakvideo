import argparse
import scipy.cluster
import scipy.spatial
import sklearn.cluster 
import pickle
import os
import numpy as np
import editdistance
import collections
import random
import math

import utils
import cluster_utils

parser = argparse.ArgumentParser()
parser.add_argument('--inputFaceFiles', type=str, required=True,
                    help='Location of pickle file outputed ny extract_face_features.py. Pass in multiple of these, one for each episode.')
parser.add_argument('--numClusters', type=int, default=-1,
                    help='The k for k-means')
parser.add_argument('--method', type=str, required=True,
                    help='Method for clustering. One of [kmeans, dbscan, spectral, agglomerative]')
args = parser.parse_args()

random.seed(1234)
np.random.seed(1234)

def processCharactersFile(charactersFile, offset):
    timeToChars = {}
    with open(charactersFile, 'r') as f:
        f.readline()
        for line in f:
            if line.count(',') != 2:
                continue
            time, name, count = line.strip().split(',')
            count = int(count)
            if len(time) == 0:
                continue
            if time in timeToChars:
                timeToChars[time].append((name, count))
            else:
                timeToChars[time] = [(name, count)]

    outputPairs = []
    for timeAsString in timeToChars.keys():
        if ':' not in timeAsString:
            print(timeAsString)
        hour, minute = timeAsString.split(':')

        # Minus 1 is because we want to start at hour=0
        timeMinutes = (float(hour)-1) * 60 + float(minute)
        timeMinutes -= offset
        timeSeconds = 60 * timeMinutes
        outputPairs.append((timeSeconds, timeToChars[timeAsString])) 
    outputPairs = sorted(outputPairs, key= lambda x: x[0])
    return outputPairs

def charactersAtTimeT(targetTime, charactersByTime):
    for timeSeconds, characters in charactersByTime:
        if timeSeconds >= targetTime:
            return characters
    raise Exception('There should be some characters Tweeted about at every possible time in the episode')

def standardizeName(charactersList, targetChar):
    for char in charactersList:
        if editdistance.eval(targetChar, char) <= 2:
            return (char, count)
    return targetChar

if __name__ == '__main__':
    '''Clusters all of the faces from one episode and visualizes the resulting clusters.'''
    allFaces = []
    with open(args.inputFaceFiles, 'r') as f:
        f.readline()
        for line in f:
            name, offset, faceFile, charactersFile, cutsFile= line.strip().split('\t')
            offset = float(offset)
            print('Reading in faces for %s...' %(name))

            charactersForEp = processCharactersFile(charactersFile, offset)
            facesForEp = utils.pickleToFaces(faceFile)
            cutsForEp = utils.readCuts(cutsFile)

            print('...retrieved %d faces' % (len(facesForEp)))

            for faceList in facesForEp.values():
                allFaces.extend(faceList)
                for face in faceList:
                    faceTime = face.getTime(cutsForEp)
                    charactersForFace = charactersAtTimeT(faceTime, charactersForEp)
                    face.characters = charactersForFace

    allFaceReps = list(face.rep for face in allFaces)

    numFaces = len(allFaces)
 
    print('Starting clustering with method: %s...' %(args.method))
    if args.method == 'kmeans':
        centroids, labels = scipy.cluster.vq.kmeans2(allFaceReps, k=args.numClusters)
        numClusters=args.numClusters
    elif args.method == 'spectral':
        cluster = sklearn.cluster.SpectralClustering(n_clusters=args.numClusters, affinity='rbf')
        sp = cluster.fit(allFaceReps)
        labels = sp.labels_
        numClusters = args.numClusters

        # forJoao = {'affinity' : sp.affinity_matrix_,
                   # 'faces' : list(f.serialize() for f in allFaces)}
        # with open('affinity.pkl', 'wb') as f:
          # pickle.dump(forJoao, f)
    elif args.method == 'agglomerative':
        cluster = sklearn.cluster.AgglomerativeClustering(n_clusters=args.numClusters, linkage='ward')
        ag = cluster.fit(allFaceReps)
        labels = ag.labels_
        numClusters = args.numClusters
    elif args.method == 'dbscan':
        db = sklearn.cluster.DBSCAN(metric='euclidean', min_samples=4).fit(allFaceReps)
        labels = db.labels_
        #dbscan figures out its own number of clusters without relying on the arg
        numClusters = len(set(x for x in labels if x != -1))
        print('%s out of %d faces have been discarded' % (len(list(x for x in labels if x == -1)), numFaces))
    else:
        raise Exception('Unsupported clustering method: %s' % (args.method))
    print('...done.')

    clustersDict = {}
    for k in xrange(0, numClusters):
        clustersDict[k] = list(allFaces[j] for j in xrange(0, numFaces) if labels[j]==k)
        print('Cluster %d has %d faces in it.' %(k, len(clustersDict[k])))

    # Remove empty clusters
    for key in clustersDict.keys():
        if clustersDict[key] == []:
            clustersDict.pop(key, None)

    if not os.path.exists('output'):
        os.mkdir('output')

    faceDim = allFaces[0].image.shape[0]

    for k in clustersDict.keys():
        characters = {}
        for face in clustersDict[k]:
            for char, count in face.characters:
                count = math.log(count)
                char = standardizeName(characters.keys(), char)
                if char in characters:
                  characters[char] += count
                else:
                  characters[char] = count
        characterCounts = collections.Counter(characters)
        topChar, topCharCount = characterCounts.most_common()[0]
        cluster_utils.visualizeOneCluster('%02d_%s' % (k, topChar), clustersDict[k], faceDim)
        print('In cluster %d, top character is "%s"' % (k, topChar))


    silhouette = sklearn.metrics.silhouette_score(allFaceReps, labels)
    print('Final silhouette coefficient = %f' % (silhouette))
    # The visualization of all the clusters in one image becomes unmanageable when the number of clusters it too high.
    # cluster_utils.visualizeAllClusters(clustersDict, faceDim, topNumToShow=25, outputPath='output.png')
