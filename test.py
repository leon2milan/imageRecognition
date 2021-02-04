import torch
import yaml
from utils import Map, _merge_a_into_b, get_cfg
from Learner import FACERecognition
from mtcnn import MTCNN
from PIL import Image
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument(
        "--img1",
        help="first img",
        default=
        '/workspace/jiangby/project/datasets/faces_glintasia/imgs/41307/1442987.jpg'
    )
    parser.add_argument(
        "--img2",
        help="second img",
        default=
        '/workspace/jiangby/project/datasets/faces_glintasia/imgs/50709/1684157.jpg'
    )
    parser.add_argument("-c",
                        "--config",
                        help="config yaml file path",
                        default='config/face_base.yaml')
    parser.add_argument('-th',
                        '--threshold',
                        help='threshold to decide identical faces',
                        default=1.54,
                        type=float)
    parser.add_argument("-tta",
                        "--tta",
                        help="whether test time augmentation",
                        action="store_true")
    parser.add_argument("-c",
                        "--score",
                        help="whether show the confidence score",
                        action="store_true")
    args = parser.parse_args()

    print('load all config')
    model_config = Map(yaml.safe_load(open(args.config)))
    cfg = _merge_a_into_b(model_config, get_cfg(), get_cfg(), [])

    device = torch.device(
        "cuda") if cfg.TEST.DEVICE == 'cuda' else torch.device("cpu")

    mtcnn = MTCNN()
    print('mtcnn loaded')

    learner = FACERecognition(cfg, True)
    learner.load_state(cfg)

    test1 = Image.open(args.img1)
    test2 = Image.open(args.img2)

    bboxes1, faces1 = mtcnn.align_multi(test1, cfg.TEST.FACE_LIMIT,
                                        cfg.TEST.MIN_FACE_SIZE)

    bboxes1, faces2 = mtcnn.align_multi(test2, cfg.TEST.FACE_LIMIT,
                                        cfg.TEST.MIN_FACE_SIZE)
    learner.infer(faces1, faces2)
