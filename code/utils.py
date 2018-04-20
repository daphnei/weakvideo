import os
import cv2
import pickle

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

def pathToName(sourceImagePath):
    return os.path.splitext(os.path.basename(sourceImagePath))[0]

def pickleToFaces(picklePath):
    '''Reads in information on the faces contained in each image in the dataset.
    
    Input: Path to a pickle file containing a list of list of serialized faces.
        Each sublist corresponds to one image from the dataset.

    Returns:
        A list of lists of Face objects.
    '''
    with open(picklePath, 'rb') as f:
        data = pickle.load(f)

    faceOutput = []
    for imageData in data:
        faceOutput.append(list(Face(fromDict=faceDict) for faceDict in imageData))
    return faceOutput


