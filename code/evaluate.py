import collections
import random
import re
import itertools

def loadGroundTruthFile(gtFile):
    output = {}
    with open(gtFile, 'r') as f:
        f.readline()
        for line in f:
            cutName, faceIndex, label = line.strip().split('\t')  
            faceIndex = int(faceIndex)

            assert (cutName, faceIndex) not in output

            if 'unknown' in label or 'null' in label:
                label = 'unknown'

            output[(cutName, faceIndex)] = label
    return output

def stupidBaselines(gtFile, frequencyFiles):
    frequencies = {}
    for idx, fpath in enumerate(frequencyFiles):
        frequencies[idx] = collections.Counter()
        with open(fpath, 'r') as f:
            f.readline()
            for line in f:
                char, count = line.strip().split(',')
                frequencies[idx][char] += int(count)

    groundtruth = loadGroundTruthFile(gtFile)

    correctCountMajority = 0
    correctCountRandom = 0
    total = 0

    countMajByEp = collections.Counter()
    countRandByEp = collections.Counter()
    totalByEp = collections.Counter()

    for (cutName, faceIndex), character in groundtruth.items():
        match = re.search(r'ep(.*).mov.*', cutName)
        ep_index = int(match.group(1)) - 1
        
        predCharMajority = frequencies[ep_index].most_common(1)[0][0]
        print(predCharMajority)

        r = random.randrange(sum(frequencies[ep_index].values()))
        predCharRandom =  next(itertools.islice(frequencies[ep_index].elements(), r, None))
       
        if character != 'unknown':
            if character == predCharMajority:
                correctCountMajority += 1
                countMajByEp[ep_index] += 1
            if character == predCharRandom:
                correctCountRandom += 1
                countRandByEp[ep_index] += 1
            totalByEp[ep_index] += 1
            total += 1

    print('Overall Majority Accuracy:' + str(float(correctCountMajority) / total))
    print('Overall Random Accuracy:' + str(float(correctCountRandom) / total))
    for c in countMajByEp:
      print('For episode ' + str(c+1))
      print('...majority accuracy: ' + str(float(countMajByEp[c]) / totalByEp[c]))
      print('...random accuracy: ' + str(float(countRandByEp[c]) / totalByEp[c]))

if __name__ == '__main__':
    gtfile = '../data/groundtruth.tsv'
    frequencyFiles = ['/nlp/users/aandy/image_to_text/episode_1_count.csv',
                      '/nlp/users/aandy/image_to_text/episode_2_count.csv',
                      '/nlp/users/aandy/image_to_text/episode_3_count.csv',
                      '/nlp/users/aandy/image_to_text/episode_4_count.csv',
                      '/nlp/users/aandy/image_to_text/episode_5_count.csv',
                      '/nlp/users/aandy/image_to_text/episode_6_count.csv',
                      '/nlp/users/aandy/image_to_text/episode_7_count.csv']
    stupidBaselines(gtfile, frequencyFiles)

def accuracy(gtFile, clusters, ignoreUnknowns=True):
    '''
    Computes the accuracy of the clustering.
 
    Inputs:
    gtFile: Path to the .tsv containing ground truth annotations
    clusters: Dictionary mapping from character name to faces corresponding
         to that character.
    ignoreUnknown: If true, do not penalize for incorrectly predicting 
         character for face annotators labeled as "unknown"
    '''
   
    groundtruth = loadGroundTruthFile(gtFile)

    correct = 0
    total = 0
    for char in clusters.keys():
        for face in clusters[char]:
            trueChar = groundtruth.get((face.sourceImageName, face.idx), None)
            if trueChar is not None:
                if not (ignoreUnknowns and trueChar == 'unknown'):
                    correct += int(trueChar == char)
                    total += 1

    print('Accuracy: ' + str(float(correct) / total))
                
   
