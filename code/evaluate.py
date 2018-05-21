def loadGroundTruthFile(gtFile):
    output = {}
    with open(gtFile, 'r') as f:
        f.readline()
        for line in f:
            cutName, faceIndex, label = line.strip().split('\t')  
            faceIndex = int(faceIndex)

            assert (cutName, faceIndex) not in output

            if 'unknown' in label:
                label = 'unknown'

            output[(cutName, faceIndex)] = label
    return output

def stupidBaselines(gtFile):
    groundtruth = loadGroundTruthFile(gtFile)
    print('Always Dany Accuracy:' )
    print('Random Accuracy:' )

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
                
   
