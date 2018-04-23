# weakvideo
This repo contains code to process a once live-streamed video and annotate it with Tweets.

# Instructions for Running Scene Segmentation
To set up PySceneDetect (detailed instructions can be found in their repo README):
```sh
pip install numpy
pip install opencv-python
git clone https://github.com/Breakthrough/PySceneDetect.git
cd PySceneDetect
python setup.py build
```

To extract scenes:
```sh
EPISODE=ep1
cd PySceneDetect
mkdir $EPISODE
cd $EPISODE
python ../scenedetect.py -i "/path/to/${EPISODE}.mov" -d content -t 30 -l -df 2 -si -co OUTPUT.csv -s STATS.csv
```

# Instructions for Running Face Detection
Install OpenFace by following the [instructions](https://cmusatyalab.github.io/openface/setup/) on their website. While OpenFace recommends using a Docker instance, I ended up installing it locally.

The script `code/extract_face_features.py` takes in a directory containing images, and extracts face vectors for each detected face in each image. Below is an example of how to run it. Note that OpenFace does **not** support python3.
```sh
OPEN_FACE_ROOT=...
python2 extract_face_features.py \
--outputDir=/data/faces/output_dir \
--networkModel="${OPEN_FACE_ROOT}/models/openface/nn4.small2.v1.t7" \
--dlibFacePredictor="${OPEN_FACE_ROOT}/models/dlib/shape_predictor_68_face_landmarks.dat \
/data/faces/input_dir
```
In the output directory, you will find:
- `.jpg`s of the aligned face images extracted from each image in the input directory.
- `face_data.pkl` which contains the vector representations of each detected face. You can download an example of this [here](seas.upenn.edu/~daphnei/data/face_data.pkl).

# Clustering Faces
To do clustering, you must run `extract_face_features.py` first to extract faces. 

## Clustering all the faces for an episode
You can run `cluster_all.py` in the following manner:
```sh
python cluster_all.py --inputFile=/path/to/face_data.pkl --numClusters=20
```

Nothing is done with the clustering results at the moment, except visualize them. You should get something like this:
![Face cluster results](images/clusters.png)

## Clustering the faces in each minute interval
You can run `cluster_timestep.py` in the following manner:
```sh
python cluster_timestep.py \
  --inputCuts=../data/ep1/CUTS.csv \
  --inputCharacters=../data/ep1/characterNames.csv \
  --inputFaces=../data/ep1/faces/face_data.pkl \
  --outputClusterDir=../data/ep1/faces_clusters
```

# Files in this repo
- `data/ep{i}/CUTS.csv`: Contains the cuts computed by PySceneDetect for Episode i
- `data/ep{i}/STATS.csv`: Contains the mean color statistics for each frame in Episode i

# Notes

## Time Offsets
The starts of the uploaded videos align with these start times of the tweets: (following Andy's convention that the first tweets are collected at 01:00:00)

- episode 1: 01:02:26
- episode 2: 01:01:14
- episode 3: 01:01:27
- episode 4: 01:01:25
- episode 5: 01:01:26
- episode 6: 01:01:33
- episode 7: 01:01:32

- Minute 1 of ep1 starts in cut 4
