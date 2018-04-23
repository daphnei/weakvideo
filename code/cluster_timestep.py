import argparse
import scipy.cluster
import scipy.spatial
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import utils

parser = argparse.ArgumentParser()
parser.add_argument('--inputFaces', type=str, required=True,
                    help='Location of pickle file outputed ny extract_face_features.py')
parser.add_argument('--inputCuts', type=str, required=True,
                    help='Location of CUTS.csv file')
parser.add_argument('--inputCharacters', type=str, required=True,
                    help='Location of characters.csv file')
parser.add_argument('--outputClusterDir', type=str, required=True,
                    help='Path to directory in which to write clustering results.')

args = parser.parse_args()

if __name__ == '__main__':
    # Dictionary from image name to list of Face objects
    faces = utils.pickleToFaces(args.inputFaces)

    # Dictionry from cut index to length in seconds 
    allCuts = utils.readCuts(args.inputCuts)

    # List of lists, each containing a set of cuts 
    bucketCuts = utils.processCutsIntoBuckets(allCuts, 60)
    
    # List of lists, each containing the characters Tweet-ed about in the ith minute. 
    characters = utils.readCharacters(args.inputCharacters)
   
    faceDim = faces.values()[0][0].image.shape[0]
    
    bucketClusters = []
    for i in xrange(len(bucketCuts)):
        cuts = bucketCuts[i]
        if i in characters:
            numCharacters = len(characters[i])
        else:
            bucketClusters.append({})
            print('For minute %d, clustering FAILED because there are 0 people expected.' % (i+1))
            continue 
        
        bucketFaceReps = []
        bucketFaces = []
        for cutIdx in cuts:
            cutName = 'ep1.mov.Scene-%03d-OUT' %(cutIdx)
            if cutName in faces:
                  listOfFaces = faces[cutName]
                  bucketFaces.extend(listOfFaces)
                  bucketFaceReps.extend(face.rep for face in listOfFaces)

        if len(bucketFaceReps) == 0:
            bucketClusters.append({})
            print('For minute %d, clustering FAILED because there are 0 faces (%d people expected).' % (i+1, numCharacters))
            continue 
        elif numCharacters > len(bucketFaceReps):
            # Prevent there from being more clusters than datapoints
            numCharacters = len(bucketFaceReps)

        print('For minute %d, clustering %d faces into %d people.' % (i+1, len(bucketFaceReps), numCharacters))
        bucketFaceReps = np.array(bucketFaceReps)
        centroids, labels = scipy.cluster.vq.kmeans2(bucketFaceReps, k=numCharacters, minit='points')
    
        clustersDict = {}
        for k in xrange(numCharacters):
            clustersDict[k] = list(bucketFaces[j] for j in xrange(0, len(bucketFaces)) if labels[j]==k)

        # Remove empty clusters
        for key in clustersDict.keys():
            if clustersDict[key] == []:
                clustersDict.pop(key, None)
        bucketClusters.append(clustersDict)

    if not os.path.exists(args.outputClusterDir):
      os.makedirs(args.outputClusterDir)

    outputFile = os.path.join(args.outputClusterDir, 'faceClusters.pkl')
    with open(outputFile, 'wb') as f:
        pickle.dump(bucketClusters, f)
        
    import pdb; pdb.set_trace()
    for idx, clusters in enumerate(bucketClusters):
        outputFile = os.path.join(args.outputClusterDir, 't_%s.png' % (idx+1))
        utils.visualizeClusters(clusters, faceDim, outputPath=outputFile)
        
