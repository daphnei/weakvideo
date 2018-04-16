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
parser.add_argument('--inputFile', type=str, required=True,
                    help='Location of pickle file outputed ny extract_face_features.py')
parser.add_argument('--numClusters', type=int, required=True,
                    help='The k for k-means')
args = parser.parse_args()

def visualizeClusters(clustersDict):
    # The number of images to show for each cluter
    topNumToShow = 10
    
    clustersShownCount = 0

    fig, axes = plt.subplots(nrows=len(clustersDict.keys()), ncols=topNumToShow, figsize=(12, 12), sharey=True)
    print('nrows is ' + str(len(clustersDict.keys())))
    # fig.suptitle('Top %d faces in each cluster' % (topNumToShow))

    # Create a figure for every 10 clusters so that figure sizes are more manageable.
    for kdx, k in enumerate(sorted(clustersDict.keys())):
        facesInCluster = clustersDict[k]
        center = centroids[k, :]
        sortedByCloseness = sorted(
                facesInCluster,
                key=lambda face: scipy.spatial.distance.euclidean(face.rep, center))
        print('k = ' + str(k))
        for i in xrange(0, topNumToShow):
            if i < len(sortedByCloseness):
                face = sortedByCloseness[i]
                axes[kdx, i].imshow(face.image)

            axes[kdx, i].get_yaxis().set_ticks([])
            axes[kdx, i].get_xaxis().set_ticks([])

        axes[kdx, 0].set_ylabel(str(k))
         
    plt.tight_layout()
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='on', labelbottom='on')
    fig.savefig('clusters.png')
    # plt.show()



if __name__ == '__main__':
    faces = utils.pickleToFaces(args.inputFile)
    
    allFaceReps = []
    allFaces = []
    for listOfFaces in faces:
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

    visualizeClusters(clustersDict)
