# weakvideo
This repo contains code to process a once live-streamed video and annotate it with Tweets.

# Instructions for Scene Segmentation
To set up PySceneDetect (detailed instructions can be found in their repo README):
```
pip install numpy
pip install opencv-python
git clone https://github.com/Breakthrough/PySceneDetect.git
cd PySceneDetect
python setup.py build
```

# Instructions for Face Detection
Install OpenFace by following the [instructions](https://cmusatyalab.github.io/openface/setup/) on their website. While OpenFace recommends using a Docker instance, I ended up installing it locally.

The script `code/extract_face_features.py` takes in a directory containing images, and extracts face vectors for each detectedface in each image. Below is an example of how to run it. Note, that OpenFace does *not* support python3.
```sh
OPEN_FACE_ROOT=...
python2 extract_face_features.py \
--outputDir=/data/faces/output_dir \
--networkModel="${OPEN_FACE_ROOT}/models/openface/nn4.small2.v1.t7" \
--dlibFacePredictor="${OPEN_FACE_ROOT}/models/dlib/shape_predictor_68_face_landmarks.dat \
/data/faces/input_dir
```

# Files in this repo
- `data/ep{i}/CUTS.csv`: Contains the cuts computed by PySceneDetect for Episode i
- `data/ep{i}/STATS.csv`: Contains the mean color statistics for each frame in Episode i
