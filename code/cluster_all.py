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
import visualization as vis
import evaluate

random.seed(1234)
np.random.seed(1234)

def modifiedKMeans(faces):
    '''Input is a list of face objects.'''
    allChars = []
    for face in faces:
        allChars.extend(c[0] for c in face.characters)
    allChars = list(set(allChars))
    # allChars = dict(zip(allChars, [0] * len(allChars)))
    
    assignments = {}
    means = {}

    # Randomly assign each image to one of its extracted names
    for face in faces:
        charName = random.choice(face.characters)[0]
        face.assignment = charName
        if charName not in assignments:
            assignments[charName] = [face]
        else:
            assignments[charName].append(face)

    numChanges = len(faces)
    stopPoint = 5

    # Used only for log statement
    idx = 0

    while numChanges > stopPoint:
        # print('Iteration %d' % (idx))
        # print('  Convergence count is at %d, need %d for convergence.' % (numChanges, stopPoint))

        # for jdx, (char, facesOfChar) in enumerate(assignments.items()):
            # print('(%d) %s: %d faces' % (jdx, char, len(facesOfChar)))
           # 
        # For each distinct name (cluster), calculate the mean of vectors assigned to that name
        for character in assignments.keys():
            facesOfChar = assignments[character]
            repsOfChar = list(face.rep for face in facesOfChar)
            means[character] = np.mean(np.array(repsOfChar), axis=0)

        # Reassign each image to the cloest mean of its extracted names.
        numChanges = 0
        newAssignments = {}
        for face in faces:
            faceRep = face.rep
            possibleCharacters = face.characterNames()

            meansFiltered = dict((character, mean) for character, mean in means.iteritems() if character in possibleCharacters) 

            meansFilteredSorted = sorted(meansFiltered.items(), 
                                 key=lambda x: np.linalg.norm(faceRep - x[1]))

            # First 0 indexes into the list of means, second gives us the 0th item in the 
            # tuple, which is the character name.
            closestChar = meansFilteredSorted[0][0]
            
            if face.assignment != closestChar:
                numChanges += 1
                face.assignment = closestChar
            if closestChar not in newAssignments:
                newAssignments[closestChar] = [face]
            else:
                newAssignments[closestChar].append(face)
        assignments = newAssignments
        idx+=1

    labels = list(allChars.index(face.assignment) for face in faces)
    print(allChars)
    num_clusters = len(assignments)
    return labels, allChars, num_clusters

def readInFaces(args):
    allFaces = []
    with open(args.inputFaceFiles, 'r') as f:
        f.readline()
        for line in f:
            name, offset, faceFile, charactersFile, cutsFile= line.strip().split('\t')
            offset = float(offset)
            print('Reading in faces for %s...' %(name))

            charactersForEp = utils.processCharactersFile(charactersFile, offset)

            facesForEp = utils.pickleToFaces(faceFile)
            cutsForEp = utils.readCuts(cutsFile)

            print('...retrieved %d faces' % (len(facesForEp)))

            for faceList in facesForEp.values():
                allFaces.extend(faceList)
                for face in faceList:
                    face.time = face.getTime(cutsForEp)
                    charactersForFace = utils.charactersAtTimeT(
                        face.time, charactersForEp, 60.*args.tweetTimeWindow)
                    face.characters = charactersForFace

    print('Founds %d faces total' % len(allFaces))
    badFaceFn = lambda f: f.tooBlurry(args.blurrinessThreshold) or f.tooSmall(args.sizeThreshold) or f.tooDark(args.darknessThreshold)
    allFaces = list(face for face in allFaces if not badFaceFn(face))
    print('Filtered out bad faces. %d faces remaning.' % (len(allFaces)))
    return allFaces

def main(args):
    '''Clusters all of the faces from one episode and visualizes the resulting clusters.'''
    if args.allFaces is None:
        allFaces = readInFaces(args)
    else:
        allFaces = args.allFaces

    allFaceReps = list(face.rep for face in allFaces)

    numFaces = len(allFaces)
    clusterNames = None

    print('Starting clustering with method: %s...' %(args.method))
    if args.method == 'kmeans':
        centroids, labels = scipy.cluster.vq.kmeans2(allFaceReps, k=args.numClusters)
        numClusters=args.numClusters
    elif args.method == 'spectral':
        cluster = sklearn.cluster.SpectralClustering(n_clusters=args.numClusters, affinity='rbf')
        sp = cluster.fit(allFaceReps)
        labels = sp.labels_
        numClusters = args.numClusters
    elif args.method == 'sparseSpectral':
        # Get the pair-wise Euclidean distance
        simMatrix = scipy.spatial.distance.cdist(allFaceReps, allFaceReps, metric='euclidean')
        # Take the RBF kernel in order to get similarity
        delta = 1
        simMatrix = np.exp( (- simMatrix ** 2) / (2. * delta ** 2))
        print('This many non-zeroes before sparsifying ' + str(np.sum(simMatrix != 0)))
        
        charStats = utils.getCharacterStats()
        # Zero out edges between faces that don't share a character. Only consider characters
        # whose frequency is higher than the median.
        for idx, face1 in enumerate(allFaces):
            for jdx, face2 in enumerate(allFaces):
                if jdx > idx:
                    continue

                edgeShouldExist = False
                sharedCharacters = set(face1.characterNames()).intersection(set(face2.characterNames()))
                for char in sharedCharacters:
                    charMedian = charStats[char][0]

                    countFace1 = list(c[1] for c in face1.characters if c[0] == char)[0]
                    countFace2 = list(c[1] for c in face2.characters if c[0] == char)[0]
                    # Normalize the counts by the interval length
                    countFace1 /= float(args.tweetTimeWindow)
                    countFace2 /= float(args.tweetTimeWindow)
                    
                    if countFace1 >= charMedian and countFace2 >= charMedian:
                        edgeShouldExist = True
                        break
                if not edgeShouldExist:
                    simMatrix[idx, jdx] = 0
                    simMatrix[jdx, idx] = 0
        # simMatrix = scipy.sparse(simMatrix)
        print('This many non-zeroes after sparsifying ' + str(np.sum(simMatrix != 0)))

        cluster = sklearn.cluster.SpectralClustering(n_clusters=args.numClusters, affinity='precomputed')
        sp = cluster.fit(simMatrix)
        labels = sp.labels_
        numClusters = args.numClusters
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
    elif args.method == 'berg2004':
        labels, clusterNames, numClusters = modifiedKMeans(allFaces)
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

    output = {}
    for k in clustersDict.keys():
        if clusterNames is None:
            characters = {}
            for face in clustersDict[k]:
                for char, count in face.characters:
                    count = math.log(count)
                    char = utils.standardizeName(characters.keys(), char)
                    if char in characters:
                      characters[char] += count
                    else:
                      characters[char] = count
            characterCounts = collections.Counter(characters)
            topChar, topCharCount = characterCounts.most_common()[0]
        else:
            topChar = clusterNames[k]
        output[topChar] = clustersDict[k]
        vis.visualizeOneCluster('%02d_%s' % (k, topChar), clustersDict[k], faceDim, saveToDisk=args.saveClusterImages)
        print('In cluster %d, top character is "%s"' % (k, topChar))

    silhouette = sklearn.metrics.silhouette_score(allFaceReps, labels)
    print('Final silhouette coefficient = %f' % (silhouette))

    evaluate.accuracy(args.groundTruthPath, output)

    return output

    # The visualization of all the clusters in one image becomes unmanageable when the number of clusters it too high.
    # vis.visualizeAllClusters(clustersDict, faceDim, topNumToShow=25, outputPath='output.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Input data parameters
    parser.add_argument('--inputFaceFiles', type=str, required=True,
                        help='Location of pickle file outputed ny extract_face_features.py. Pass in multiple of these, one for each episode.')
    parser.add_argument('--darknessThreshold', type=float, default=None,
                        help='Filter out images darker than this. (None does no filtering)')
    parser.add_argument('--sizeThreshold', type=float, default=None,
                        help='Filter out images smaller than this. (None does no filtering)')
    parser.add_argument('--blurrinessThreshold', type=float, default=None,
                        help='Filter out images blurrier than this. (None does no filtering)')
    
    # Weak annotation parameters
    parser.add_argument('--annotation', type=str, default='tweets',
                        help='Options are either "tweets" to use the character frequencies directly, or else a path to an alternative annotation file.')
    parser.add_argument('--tweetTimeWindow', type=int, default=3,
                        help='A face will be labeled with all tweets within this many minutes of its timestamp.')

    # Clustering parameters
    parser.add_argument('--numClusters', type=int, default=-1,
                        help='The k for k-means')
    parser.add_argument('--method', type=str, required=True,
                        help='Method for clustering. One of [kmeans, dbscan, spectral, agglomerative, berg200]')

    parser.add_argument('--saveClusterImages', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='If flag is set, cluster images are saved to disk. Otherwise they are just displayed.')
    parser.add_argument('--groundTruthPath', default='../data/groundtruth.tsv', type=str,
                        help='Location of groundtruth labels for evaluation of clustering.')

    parser.add_argument('--allFaces', default=None,
                        help='For use in Jupyter notebook only')
    args = parser.parse_args()
    main(args)
