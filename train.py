import torch
import yaml
from utils import Map, _merge_a_into_b, get_cfg
from Learner import FACERecognition
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-c",
                        "--config",
                        help="config yaml file path",
                        default='config/face_base.yaml')
    args = parser.parse_args()

    print('load all config')
    model_config = Map(yaml.safe_load(open(args.config)))
    cfg = _merge_a_into_b(model_config, get_cfg(), get_cfg(), [])

    device = torch.device("cuda") if cfg.MODEL.DEVICE else torch.device("cpu")

    learner = FACERecognition(cfg)

    learner.train(cfg)
