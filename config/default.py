from utils import Map

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_DEFAULT = Map()

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_DEFAULT.MODEL = Map()
_DEFAULT.MODEL.DEVICE = "cuda"
_DEFAULT.MODEL.META_ARCHITECTURE = "Baseline"

_DEFAULT.MODEL.FREEZE_LAYERS = ['']

# MoCo memory size
_DEFAULT.MODEL.QUEUE_SIZE = 8192

# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_DEFAULT.MODEL.BACKBONE = Map()

_DEFAULT.MODEL.BACKBONE.NAME = "build_resnet_backbone"
_DEFAULT.MODEL.BACKBONE.DEPTH = 50
_DEFAULT.MODEL.BACKBONE.NET_MODE = "ir_se"
_DEFAULT.MODEL.BACKBONE.FEAT_DIM = 2048
# Pretrain model path
_DEFAULT.MODEL.BACKBONE.DROP_RATIO = 0.6

# ---------------------------------------------------------------------------- #
# REID HEADS options
# ---------------------------------------------------------------------------- #
_DEFAULT.MODEL.HEADS = Map()
_DEFAULT.MODEL.HEADS.NAME = "EmbeddingHead"
# Number of identity
_DEFAULT.MODEL.HEADS.NUM_CLASSES = 0
# Embedding dimension in head
_DEFAULT.MODEL.HEADS.EMBEDDING_DIM = 0

# Classification layer type
_DEFAULT.MODEL.HEADS.CLS_LAYER = "linear"  # "arcSoftmax" or "circleSoftmax"

# Margin and Scale for margin-based classification layer
_DEFAULT.MODEL.HEADS.MARGIN = 0.15
_DEFAULT.MODEL.HEADS.SCALE = 128

# ---------------------------------------------------------------------------- #
# REID LOSSES options
# ---------------------------------------------------------------------------- #
_DEFAULT.MODEL.LOSSES = Map()
_DEFAULT.MODEL.LOSSES.NAME = ("CrossEntropyLoss",)

# Cross Entropy Loss options
_DEFAULT.MODEL.LOSSES.CE = Map()
# if epsilon == 0, it means no label smooth regularization,
# if epsilon == -1, it means adaptive label smooth regularization
_DEFAULT.MODEL.LOSSES.CE.EPSILON = 0.0
_DEFAULT.MODEL.LOSSES.CE.ALPHA = 0.2
_DEFAULT.MODEL.LOSSES.CE.SCALE = 1.0

# Focal Loss options
_DEFAULT.MODEL.LOSSES.FL = Map()
_DEFAULT.MODEL.LOSSES.FL.ALPHA = 0.25
_DEFAULT.MODEL.LOSSES.FL.GAMMA = 2
_DEFAULT.MODEL.LOSSES.FL.SCALE = 1.0

# Triplet Loss options
_DEFAULT.MODEL.LOSSES.TRI = Map()
_DEFAULT.MODEL.LOSSES.TRI.MARGIN = 0.3
_DEFAULT.MODEL.LOSSES.TRI.NORM_FEAT = False
_DEFAULT.MODEL.LOSSES.TRI.HARD_MINING = True
_DEFAULT.MODEL.LOSSES.TRI.SCALE = 1.0

# Circle Loss options
_DEFAULT.MODEL.LOSSES.CIRCLE = Map()
_DEFAULT.MODEL.LOSSES.CIRCLE.MARGIN = 0.25
_DEFAULT.MODEL.LOSSES.CIRCLE.GAMMA = 128
_DEFAULT.MODEL.LOSSES.CIRCLE.SCALE = 1.0

# Cosface Loss options
_DEFAULT.MODEL.LOSSES.COSFACE = Map()
_DEFAULT.MODEL.LOSSES.COSFACE.MARGIN = 0.25
_DEFAULT.MODEL.LOSSES.COSFACE.GAMMA = 128
_DEFAULT.MODEL.LOSSES.COSFACE.SCALE = 1.0

# Path to a checkpoint file to be loaded to the model. You can find available models in the model zoo.
_DEFAULT.MODEL.WEIGHTS = ""

# Values to be used for image normalization
_DEFAULT.MODEL.PIXEL_MEAN = [0.485*255, 0.456*255, 0.406*255]
# Values to be used for image normalization
_DEFAULT.MODEL.PIXEL_STD = [0.229*255, 0.224*255, 0.225*255]

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_DEFAULT.INPUT = Map()
# Size of the image during training
_DEFAULT.INPUT.SIZE_TRAIN = [256, 128]
# Size of the image during test
_DEFAULT.INPUT.SIZE_TEST = [256, 128]

# Random probability for image horizontal flip
_DEFAULT.INPUT.DO_FLIP = True
_DEFAULT.INPUT.FLIP_PROB = 0.5

# Value of padding size
_DEFAULT.INPUT.DO_PAD = True
_DEFAULT.INPUT.PADDING_MODE = 'constant'
_DEFAULT.INPUT.PADDING = 10

# Random color jitter
_DEFAULT.INPUT.CJ = Map()
_DEFAULT.INPUT.CJ.ENABLED = False
_DEFAULT.INPUT.CJ.PROB = 0.5
_DEFAULT.INPUT.CJ.BRIGHTNESS = 0.15
_DEFAULT.INPUT.CJ.CONTRAST = 0.15
_DEFAULT.INPUT.CJ.SATURATION = 0.1
_DEFAULT.INPUT.CJ.HUE = 0.1

# Random Affine
_DEFAULT.INPUT.DO_AFFINE = False

# Auto augmentation
_DEFAULT.INPUT.DO_AUTOAUG = False
_DEFAULT.INPUT.AUTOAUG_PROB = 0.0

# Augmix augmentation
_DEFAULT.INPUT.DO_AUGMIX = False
_DEFAULT.INPUT.AUGMIX_PROB = 0.0

# Random Erasing
_DEFAULT.INPUT.REA = Map()
_DEFAULT.INPUT.REA.ENABLED = False
_DEFAULT.INPUT.REA.PROB = 0.5
_DEFAULT.INPUT.REA.VALUE = [0.485*255, 0.456*255, 0.406*255]
# Random Patch
_DEFAULT.INPUT.RPT = Map()
_DEFAULT.INPUT.RPT.ENABLED = False
_DEFAULT.INPUT.RPT.PROB = 0.5

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_DEFAULT.DATASETS = Map()
# List of the dataset names for training
_DEFAULT.DATASETS.NAMES = "emore"
# Combine trainset and testset joint training
_DEFAULT.DATASETS.COMBINEALL = False
_DEFAULT.DATASETS.FOLDER = "/workspace/jiangby/project/datasets/faces_glintasia"

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_DEFAULT.DATALOADER = Map()
# Number of instance for each person
_DEFAULT.DATALOADER.NUM_INSTANCE = 4
_DEFAULT.DATALOADER.NUM_WORKERS = 4

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_DEFAULT.SOLVER = Map()

# AUTOMATIC MIXED PRECISION
_DEFAULT.SOLVER.FP16_ENABLED = False

# Optimizer
_DEFAULT.SOLVER.OPT = "Adam"

_DEFAULT.SOLVER.MAX_EPOCH = 120

_DEFAULT.SOLVER.BASE_LR = 3e-4
_DEFAULT.SOLVER.BIAS_LR_FACTOR = 1.
_DEFAULT.SOLVER.HEADS_LR_FACTOR = 1.

_DEFAULT.SOLVER.MOMENTUM = 0.9
_DEFAULT.SOLVER.NESTEROV = True

_DEFAULT.SOLVER.WEIGHT_DECAY = 0.0005
_DEFAULT.SOLVER.WEIGHT_DECAY_BIAS = 0.

# Multi-step learning rate options
_DEFAULT.SOLVER.SCHED = "MultiStepLR"

_DEFAULT.SOLVER.DELAY_EPOCHS = 0

_DEFAULT.SOLVER.GAMMA = 0.1
_DEFAULT.SOLVER.STEPS = [30, 55]

# Cosine annealing learning rate options
_DEFAULT.SOLVER.ETA_MIN_LR = 1e-7

# Warmup options
_DEFAULT.SOLVER.WARMUP_FACTOR = 0.1
_DEFAULT.SOLVER.WARMUP_ITERS = 1000
_DEFAULT.SOLVER.WARMUP_METHOD = "linear"

# Backbone freeze iters
_DEFAULT.SOLVER.FREEZE_ITERS = 0

# FC freeze iters
_DEFAULT.SOLVER.FREEZE_FC_ITERS = 0


# SWA options
# _DEFAULT.SOLVER.SWA = Map()
# _DEFAULT.SOLVER.SWA.ENABLED = False
# _DEFAULT.SOLVER.SWA.ITER = 10
# _DEFAULT.SOLVER.SWA.PERIOD = 2
# _DEFAULT.SOLVER.SWA.LR_FACTOR = 10.
# _DEFAULT.SOLVER.SWA.ETA_MIN_LR = 3.5e-6
# _DEFAULT.SOLVER.SWA.LR_SCHED = False

_DEFAULT.SOLVER.CHECKPOINT_PERIOD = 20

# Number of images per batch across all machines.
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_DEFAULT.SOLVER.IMS_PER_BATCH = 64

# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_DEFAULT.TEST = Map()

_DEFAULT.TEST.EVAL_PERIOD = 1000
_DEFAULT.TEST.SAVE_PERIOD = 10000
_DEFAULT.TEST.SHOW_PERIOD = 100

# Number of images per batch in one process.
_DEFAULT.TEST.METRIC = "cosine"
_DEFAULT.TEST.MODE = "face"
_DEFAULT.TEST.ROC_ENABLED = False
_DEFAULT.TEST.FLIP_ENABLED = False
_DEFAULT.TEST.FACE_LIMIT = 10
_DEFAULT.TEST.MIN_FACE_SIZE = 30
_DEFAULT.TEST.DEVICE = "cpu"

# Benchmark different cudnn algorithms.
# If input images have very different sizes, this option will have large overhead
# for about 10k iterations. It usually hurts total time, but can benefit for certain models.
# If input images have the same or similar sizes, benchmark is often helpful.
_DEFAULT.CUDNN_BENCHMARK = False


_DEFAULT.CHECKPOINT = Map()
_DEFAULT.CHECKPOINT.SAVE_DIR = "checkpoints/"
_DEFAULT.CHECKPOINT.RESTORE = False
_DEFAULT.CHECKPOINT.RESTORE_MODEL = ""
