# Facenet

Requires:
* Linux
* Anaconda
* Python

./Facenet/ contains :
* Facenet from https://github.com/davidsandberg/facenet
* Model from https://drive.google.com/file/d/1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-/view
* environment.yml
* fn_compare.py script adapted from original compare.py (modifications allows comparing images from path 1 to images in path2 and copying mathcing scores list in results.csv)

[1] create conda environment 
```
cd facenet/
conda env create -f environment.yml
conda activate Facenet2.7
```

[2] Run fn_compare.py 
```
cd src/
python fn_compare.py ~/Data/fold1/ ~/Data/fold2/ ~/results.csv
```
