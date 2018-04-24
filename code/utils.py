import os
import cv2
import pickle
import pandas
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

class Face:
    def __init__(self, sourceImagePath=None, idx=None, bb=None, image=None, fromDict=None):
        """Represents one aligned face extracted from an image.
        
        Inputs:
        sourceImagePath: Path to the image this face was found in
        idx: A unique index for this Face within the image it belongs to
        bb: The bounding box around this face in the image
        image: An aligned version of the contents of the bounding box in the source image
        """
        if fromDict is not None:
            self.sourceImageName = fromDict['sourceImageName']
            self.idx = fromDict['idx']
            self.bbTopLeft = fromDict['bbTopLeft']
            self.bbBottomRight = fromDict['bbBottomRight']
            self.image = fromDict['image']
            self.rep = fromDict['rep']
        else:
            self.sourceImageName = pathToName(sourceImagePath) 
            self.idx = idx
            self.bbTopLeft = (bb.left(), bb.top())
            self.bbBottomRight = (bb.right(), bb.bottom())
            self.image = image

    def saveImage(self, outputDir):
        """Saved aligned face image as .jpg on disk.
        """
        interDir = os.path.join(outputDir, self.sourceImageName)
        if not os.path.exists(interDir):
            os.makedirs(interDir)

        write_path = os.path.join(interDir, '%02d.jpg' % (self.idx))
        bgrImage = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(write_path, bgrImage)
        
    def computeRep(self, facenet):
        """Runs a forward pass of NN to get a vector representation of the aligned face image.
        """
        self.rep = facenet.forward(self.image)


    def serialize(self):
        return {
          'sourceImageName': self.sourceImageName,
          'idx': self.idx,
          'bbTopLeft': self.bbTopLeft,
          'bbBottomRight': self.bbBottomRight,
          'image': self.image,
          'rep': self.rep}

def pickleToFaces(picklePath):
    '''Reads in information on the faces contained in each image in the dataset.
    
    Input: Path to a pickle file containing a list of list of serialized faces.
        Each sublist corresponds to one image from the dataset.

    Returns:
        A list of lists of Face objects.
    '''
    with open(picklePath, 'rb') as f:
        data = pickle.load(f)

    faceOutput = {}
    for name, imageData in data.items():
        faceOutput[name] = list(Face(fromDict=faceDict) for faceDict in imageData)
    return faceOutput

def pathToName(sourceImagePath):
    '''Converts a path into a name based on the filename stripped of its extension.'''
    return os.path.splitext(os.path.basename(sourceImagePath))[0]

def readCuts(cutsPath):
    '''Reads a cuts csv file into a ditionary mapping cut index to the duration of that cut in seconds.
    '''
    data = pandas.read_csv(cutsPath, sep=',')
    return dict(zip(data['Scene Number'], data['Length (seconds)']))

def readCharacters(charactersPath):
    '''Reads a chararcter csv file into a list of lists, each containing
       the characters Tweeted about in the ith minute. 
    '''

    f = open(charactersPath)
    print(f.readline())

    charactersByTime = {}
    for row in f:
        time, character, count = row.strip().split(',')
        # TODO: This is hard-coded for episode 1, needs to be fixed.
        time = int(time[-2:]) - 3

        if time not in charactersByTime:
            charactersByTime[time] = [character]
        else:
            charactersByTime[time].append(character)

    charactersByTimeList = []
    for timeIdx in range(max(charactersByTime.keys())):
        if timeIdx in charactersByTime:
            charactersByTimeList.append(charactersByTime[timeIdx])
        else:
            charactersByTimeList.append([])
    return charactersByTimeList

def processCutsIntoBuckets(cuts, lengthInMinutes):
    '''Buckets the cuts into groups of those falling into each time interval.

    Inputs:
    cuts: a dictionary for cut index to length of the cutin seconds
    lengthInMinutes: the time interval to use for bucketing

    Returns:
    List of lists, each containing the set of cuts corrpesonding to the ith time interval.
    '''
    lengthInSeconds = lengthInMinutes * 60.0

    timeSoFar = 0

    bucketCount  = 1
    cutsInBucket = []

    buckets = []
    
    for cutIdx, cutLength in cuts.iteritems():
        cutsInBucket.append(cutIdx)
         
        timeSoFar += cutLength
        while timeSoFar >= bucketCount * lengthInSeconds:
            buckets.append(cutsInBucket)
            cutsInBucket = [cutIdx]
            bucketCount+=1

    return buckets

def processCharactersIntoBuckets(characters, lengthInMinutes):
    '''Buckets the characters into groups of thos mentioned in each time interval.

    Inputs:
    characters: list of lists of characters, where list i contains the characters mentioned in the ith minute.
    lengthInMinutes: the time interval to use for bucketing

    Returns:
    List of lists, each containing the set of characters mentioned in the ith time interval.
    '''
    assert int(lengthInMinutes) == lengthInMinutes, 'Only support integer time inervals (in minutes) for now.'

    allBuckets = []

    charsInBucket = None

    for minuteIdx in xrange(0, len(characters)):
        if minuteIdx % lengthInMinutes == 0:
            if charsInBucket is not None:
                allBuckets.append(charsInBucket)
            charsInBucket = set()

        charList = characters[minuteIdx]
        charsInBucket = charsInBucket.union(set(charList))

    return allBuckets

        
def visualizeClusters(clustersDict, faceDim, topNumToShow=20, outputPath=None):
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

    print('nrows is ' + str(len(clustersDict.keys())))

    blackLine = np.zeros([3, faceDim * topNumToShow, 3], dtype=np.uint8)
    outputImage = np.array(blackLine, copy=True)

    for kdx, k in enumerate(sorted(clustersDict.keys())):
        facesInCluster = clustersDict[k]
        # center = centroids[k, :]
        # sortedByCloseness = sorted(
                # facesInCluster,
                # key=lambda face: scipy.spatial.distance.euclidean(face.rep, center))
        sortedByCloseness = facesInCluster

        print('k = ' + str(k))
        
        numFacesToShow = min(len(sortedByCloseness), topNumToShow)
        rowImage = np.concatenate(list(face.image for face in sortedByCloseness[:numFacesToShow]), axis=1)
        extraBlack = np.zeros([faceDim, (topNumToShow - numFacesToShow) * faceDim, 3], dtype=np.uint8) 
        rowImage = np.concatenate([rowImage, extraBlack], axis=1)
        
        outputImage = np.concatenate([outputImage, rowImage, blackLine], axis=0)
    plt.imshow(outputImage)
           
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
    plt.tight_layout()

    if outputPath is None:
        plt.show()
    else:
        plt.savefig(outputPath, dpi=300)



