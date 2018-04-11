"""Uses OpenFace to extract faces and their vector representations from all images in a directory."""

import time

start = time.time()

import argparse
import cv2
import itertools
import os
import glob
import pickle

import numpy as np
np.set_printoptions(precision=2)

import openface
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

parser = argparse.ArgumentParser()

parser.add_argument('imageDir', type=str, help="Directory containing the input images.")
parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
parser.add_argument('--outputDir', type=str,
                    help="Where to output detected and transformed faces.", default='output')
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()

class Face:
    def __init__(self, sourceImagePath, idx, bb, image):
        """Represents one aligned face extracted from an image.
        
        Inputs:
        sourceImagePath: Path to the image this face was found in
        idx: A unique index for this Face within the image it belongs to
        bb: The bounding box around this face in the image
        image: An aligned version of the contents of the bounding box in the source image
        """
        self.sourceImageName = os.path.splitext(os.path.basename(sourceImagePath))[0]
        self.idx = idx
        self.bb = bb
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

def getAlignedFaces(alignDlib, imagePath, imgDim):
    """Detects faces in the specificed image

    Inputs:
    alignDlib: the dlib model
    imagePath: Path to jpg or other image file
    imgDim: What dimension to resize cropped faces to

    Returns:
    A list of Face objects, one for each detected face. 
    """
    
    bgrImg = cv2.imread(imagePath)
    if bgrImg is None:
        raise Exception("ERROR: unable to load image: {}".format(imgPath))

    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    start = time.time()
    bbs = alignDlib.getAllFaceBoundingBoxes(rgbImg)
    if bbs is None or len(bbs) == 0:
        print("Warning: unable to find a face in: {}".format(imagePath))
        return []
    print("  + Face detection took {} seconds.".format(time.time() - start))

    start = time.time()
    faces = []
    for idx, bb in enumerate(bbs):
        alignedFace = alignDlib.align(imgDim, rgbImg, bb,
                                landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if alignedFace is None:
            print("Warning: Unable to align face {} in image: {}".format(idx, imagePath))

        faces.append(Face(imagePath, idx, bb, alignedFace))
    print("  + Face alignment took {} seconds.".format(time.time() - start))
    return faces


def saveAlignedFaces(alignedFaces, outputDir):
    """ For each aligned face in the list, save a .jpg.

    Input:
    alignedFaces: list of face objects
    outputDir: Where to save to
    """
    for face in alignedFaces:
        face.saveImage(outputDir)


def getReps(alignedFaces, facenet):
    """ Compute the vector representation of each face in the list of faces.
    
    Results are saved into each Face object rather than returned.

    Input:
    alignedFaces: list of face objects
    facenet: Torch model that takes in image matrix and outputs vector representation.
    """
    start = time.time()
    for face in alignedFaces:
        face.computeRep(facenet)
    print("  + OpenFace forward passes took {} seconds.".format(time.time() - start))

if __name__ == '__main__':
    print("Loading the dlib model")
    alignDlib = openface.AlignDlib(args.dlibFacePredictor)
    facenet = openface.TorchNeuralNet(args.networkModel, args.imgDim)

    # Make sure output dir exists.
    if not os.path.exists(args.outputDir):
        os.makedirs(args.outputDir)

    if not os.path.exists(args.imageDir):
        raise Exception('Unable to locate directory: %s' % (args.imageDir))

    allFaces = {}
    for imagePath in glob.glob(os.path.join(args.imageDir, '*.jpg')):
        print('Processing: %s' % (imagePath))

        alignedFaces = getAlignedFaces(alignDlib, imagePath, args.imgDim)
        saveAlignedFaces(alignedFaces, args.outputDir)
        reps = getReps(alignedFaces, facenet)
        allFaces[imagePath] = (alignedFaces)

    with open(os.path.join(args.outputDir, 'face_data.pkl'), 'wb') as f:
      pickle.dump(allFaces, f)

