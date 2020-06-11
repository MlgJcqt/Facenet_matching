# Facenet

Requires:
* Linux
* Anaconda
* Python

[1] Get facenet from https://github.com/davidsandberg/facenet
```
git clone https://github.com/davidsandberg/facenet
```

[2] Get model from https://drive.google.com/file/d/1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-/view
and extract to ~/facenet/data/model/20180402-114759.pb

[3] Put environment.yml and requirements.txt in ~/facenet/

[4] create ~/facenet/Data/images1/ and ~/facenet/Data/images2/ with images to compare

[5] set conda environment 
```
cd facenet/
conda env create -f environment.yml
conda activate Facenet2.7
```

[6] Run fn_compare.py 
```
cd src/
python fn_compare.py ~/facenet/Data/images1/ ~/facenet/Data/images2/ ~/results.csv
```
*fn_compare.py script adapted from original compare.py (modifications allows comparing images from path 1 to images in path2 and copying mathcing scores list in results.csv)
