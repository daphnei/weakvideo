import os
import cv2
import pickle
import re
import pandas
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
            self.rep = fromDict.get('rep', None)
            self.characters = fromDict.get('characters', None)
        else:
            self.sourceImageName = pathToName(sourceImagePath) 
            self.idx = idx
            self.bbTopLeft = (bb.left(), bb.top())
            self.bbBottomRight = (bb.right(), bb.bottom())
            self.image = image
            self.characters = None
            self.rep = None

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
        return self.rep

    def getTime(self, sceneTimes):
        match = re.match(r'ep.*.mov.Scene-(\d+)-(IN|OUT)', self.sourceImageName)
        if match is None:
            return None
        else:
            sceneId = int(match.groups()[0])
            imType = match.groups()[1]
            if imType == 'OUT':
                # If this frame was at the end of the scene, we instead want to mark it with
                # the timestamp of the beginning of the next scene. 
                sceneId += 1
            return sceneTimes[sceneId]


    def serialize(self):
        return {
          'sourceImageName': self.sourceImageName,
          'idx': self.idx,
          'bbTopLeft': self.bbTopLeft,
          'bbBottomRight': self.bbBottomRight,
          'image': self.image,
          'rep': self.rep,
          'characters': self.characters}

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

def readCharacters(charactersPath, episodeDuration=60):
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
    for timeIdx in range(episodeDuration):
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
    '''Buckets the characters into groups of those mentioned in each time interval.

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

def processCharactersFile(charactersFile, offset):
    ''' Reads the characters file into a data structure.

    Input: 
    charactersFile: Path to a chararacters.csv, giving the characters
            mentioned in each minute.
    offset: Offset in seconds between when the Tweet stream starts and
        when the video (on which the face timestamps are based) starts.

    Returns a list of tuples. The first element is a time (seconds) in
    the video timespace. The second element is a list of tuples where
    the first element is a character name and the second element is the
    character frequency. The characer lists are sorted in increasing 
    order of time. 
    '''

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

def charactersAtTimeT(targetTime, charactersByTime, intervalLength=180):
    ''' Returns all of the characters who were mentioned within the
        specified time interval following the targetTime.

        For example, if targetTime is 5.0 and intervalLength is 120,
        returns all of the characters mentioned in the two minutes 
        folowing the 5 second mark of the video. 
        
        Inputs:
        targetTime: A time in seconds (from the beginning of the video).
        charactersByTime: List of ordered (increasig by time) tuples.
            First element is time, second element is a tuple of character and
            frequency of mentions.
        intervalLength: How many seconds after the targetTime to grab characters
                mentions from.

        Returns: A list of (character name, frquency) tuples.
    '''
    mentionedCharacters = []
    for timeSecondsOfMention, characters in charactersByTime:
        if timeSecondsOfMention >= targetTime:
            mentionedCharacters.extend(characters)
        if timeSecondsOfMention >= targetTime + intervalLength:
            break

    output = {}
    for char, count in mentionedCharacters:
      if char not in output:
        output[char] = count
      else:
        output[char] += count
    return list(output.iteritems())


def standardizeName(charactersList, targetChar):
    for char in charactersList:
        if editdistance.eval(targetChar, char) <= 2:
            return (char, count)
    return targetChar

      
