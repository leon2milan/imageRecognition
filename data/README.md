### Train

1. Install `MXNet` with GPU support (Python 3.X).

```
pip install mxnet-cu101 # which should match your installed cuda version
```

2. Download the training set (`MS1M-Arcface`) and place it in *`/workspace/jiangby/project/datasets/faces_glintasia`*. Each training dataset includes at least following 6 files:

```Shell
    faces_emore/
       train.idx
       train.rec
       property
       lfw.bin
       cfp_fp.bin
       agedb_30.bin
```

3. Process
`python preprocess.py`

