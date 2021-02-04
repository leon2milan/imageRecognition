# Image Recognition
本项目用于构建图像类识别以及验证。

# 环境
`conda activate darwinml_ve_2.3`

# 数据收集

## 人脸数据

- [x] 1. 下载格林深瞳开源数据集 (共计9.4万ID，280万张图片)。

 - [更多数据可以查看](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo)

 - 此外格林深瞳数据使用mxnet存储，可以使用`python data/preprocess.py` 进行处理。 （其中age30， cfp-fp， lfw等为人脸的测试数据）
 - 若使用格林深瞳数据 需要 `pip install -r requirments.txt`

- [x] 2. 更新环境变量 `export PYTHONPATH=$model/facenet/src:$PYTHONPATH` （`$model` 代表本代码所在目录）

- [x] 3. 运行如下命令crop出人脸所在区域

`python facenet/src/align/align_dataset_mtcnn.py ../datasets/asian_face/raw/ ../datasets/asian_face/imgs/ --image_size 112`

如果识别对别其他类型数据， 需要得到大小为[112, 112]的图像， align后的图像保存在`$data/imgs`下

- 也可以使用其他align model对齐人脸

## 公章数据

- [ ] 1. 使用fake 公章 + 真实公章数据。 抽取20% fake 公章 + 真实公章数据作为test 数据。

* 需要大量各类型数据

* fake 公章最好支持各种真实场景，如不同底纹， 不同来源（拍照， 影印等）， 不同角度等

- [ ] 2. 需要公章目标检测模型检测公章

- [ ] 3. 对其公章模型

- [x] 4. 生成测试数据的pair文件

* 更新环境变量 `export PYTHONPATH=$model:$PYTHONPATH`

* 在`data`目录下运行`python generate_pairs.py`。 需要修改代码中的测试数据集的路径。

* test 数据集最好人工review，且固定不变。因为需要根据test 数据集生成best thresho

## 数据统一格式

数据统一格式如下（`$data` 代表数据所在目录），

```


$data/raw/
       ---> id1
            ---> id1_1.jpg
       ---> id2
            ---> id2_1.jpg
       ---> id3
            ---> id3_1.jpg
            ---> id3_2.jpg
```

# 训练模型
## 参数配置
全部参数设置位于`config/default.py`,   针对新模型，只需要新建`config/XX.yaml`文件，并设置对应参数即可。同时需要修改`train.py`中yaml 文件路径

## 训练
`python train.py --config config/XX.yaml`

## 模型结构
### 数据清洗
- [ ] 数据增强

### BACKBONE
- [x] resnet 50/ 100
- [x] mobilenet

### METRIC
- [X] ARC SOFTMAX
- [X] COS SOFTMAX
- [X] CIRCLE SOFTMAX

### LOSS
- [x] FOCAL LOSS
- [x] CIRCLE LOSS
- [x] TRIPLET LOSS

### LOSS COMBINE
- [x] average
- [ ] 自定义系数

### OPTIMIZATION
- [X] SGD
- [X] ADAM
- [ ] OTHER

### ADAPTIVE LR
- [X] CosineAnnealingLR
- [ ] OTHER

# 模型预测

`python test.py --config config/XX.yaml --threshold`

注: 代码中包含MTCNN的模型对齐。 如果是非人脸类其他数据， 需要修改代码。

# 实时对比

- [ ]  图像Verification
后续加入机器学习分类模型， 如SVM判断是否为同一人。

- [ ]  图像Recognition
需要将imagebank 中的图像加载至内存中。 imagebank数据量如果过大， 推荐使用faiss进行检索。

# Reference
[facenet & triplet loss](https://www.cv-foundation.org/openaccess/content_cvpr_2015/app/1A_089.pdf)  
[focal loss](https://arxiv.org/pdf/1708.02002.pdf)    
[circle loss](https://arxiv.org/abs/2002.10857)   
[cosface](https://arxiv.org/abs/1801.09414)  
[arcface](https//arxiv.org/abs/1801.07698)  