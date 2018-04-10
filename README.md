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

# Files in this repo
- `data/ep{i}/CUTS.csv`: Contains the cuts computed by PySceneDetect for Episode i
- `data/ep{i}/STATS.csv`: Contains the mean color statistics for each frame in Episode i
