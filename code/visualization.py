import pickle
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


def visualizeImageGrid(rowsOfFaces, faceDim, numFacesPerRow, outputPath=None):
    '''Creates a visual consisting of a grid of face images

    Inputs:
    rowsOfFaces: A list of lists of Face objects. Each list corresponds to the 
        faces which should appear in one row.
    faceDim: The size of a face in pixels
    numFacesPerRow: The max number of images to show in a row. Each list in 
        rowsOfFaces will be truncated to this number.
    outputPath: Where to save the figure to. If none, figure is just shown, not saved.
    '''

    blackLine = np.zeros([3, faceDim * numFacesPerRow, 3], dtype=np.uint8)
    outputImage = np.array(blackLine, copy=True)
    for facesForRow in rowsOfFaces:
        facesForRow = facesForRow[:numFacesPerRow]

        rowImage = np.concatenate(list(face.image for face in facesForRow), axis=1)
        extraBlack = np.zeros([faceDim, (numFacesPerRow - len(facesForRow)) * faceDim, 3], dtype=np.uint8) 
        rowImage = np.concatenate([rowImage, extraBlack], axis=1)

        outputImage = np.concatenate([outputImage, rowImage, blackLine], axis=0)

    if outputPath is None:
        plt.show()
        plt.imshow(outputImage)
               
        plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
        plt.tight_layout()
        plt.show()
    else:
        im = Image.fromarray(outputImage)
        im.save(outputPath)

def visualizeOneCluster(clusterName, faces, faceDim):
    numFacesPerRow = 30

    rowsOfFaces = []
    faceIdx = 0
    while faceIdx < len(faces):
        facesForRow = faces[faceIdx:faceIdx+numFacesPerRow]
        rowsOfFaces.append(facesForRow)
        faceIdx += numFacesPerRow
    outputPath = os.path.join('output', clusterName + '.png')
    visualizeImageGrid(rowsOfFaces, faceDim, numFacesPerRow, outputPath=outputPath)

def visualizeAllClusters(clustersDict, faceDim, topNumToShow=20, outputPath=None):
    '''Given a clustering of Face objcets, generates a grid of the faces from each cluster.

    # TODO: Implement face sorting
    Faces are ordered in each row from closest to cluster center (left) to furthest (right).

    Inputs:
    clustersDict: Maps from cluster ID to list of Face objects
    faceDim: The size of a face in pixels
    topNumToShow: How many faces from each cluster to show.
    outputPath: Where to save the figure to. If none, figure is just shown, not saved.
    '''

    clustersShownCount = 0

    rowsOfFaces = []
    for kdx, k in enumerate(sorted(clustersDict.keys())):
        facesInCluster = clustersDict[k]
        # center = centroids[k, :]
        # sortedByCloseness = sorted(
                # facesInCluster,
                # key=lambda face: scipy.spatial.distance.euclidean(face.rep, center))
        sortedByCloseness = facesInCluster
        rowsOfFaces.append(sortedByCloseness)
    visualizeImageGrid(rowsOfFaces, faceDim, topNumToShow)


    
