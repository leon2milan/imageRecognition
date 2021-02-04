#! encoding: utf-8

import os
import random
import yaml
from utils import Map, _merge_a_into_b, get_cfg


class GeneratePairs:
    """
    Generate the pairs.txt file that is used for training face classifier when calling python `src/train_softmax.py`.
    Or others' python scripts that needs the file of pairs.txt.
    Doc Reference: http://vis-www.cs.umass.edu/lfw/README.txt
    """
    def __init__(self, data_dir, pairs_filepath, img_ext):
        """
        Parameter data_dir, is your data directory.
        Parameter pairs_filepath, where is the pairs.txt that belongs to.
        Parameter img_ext, is the image data extension for all of your image data.
        """
        self.data_dir = data_dir
        self.pairs_filepath = pairs_filepath
        self.img_ext = img_ext

    def generate(self):
        self._generate_matches_pairs()
        self._generate_mismatches_pairs()

    def _generate_matches_pairs(self):
        """
        Generate all matches pairs
        """
        for name in os.listdir(self.data_dir):
            if name == ".DS_Store":
                continue

            a = []
            for file in os.listdir(self.data_dir + name):
                if file == ".DS_Store":
                    continue
                a.append(file)

            with open(self.pairs_filepath, "a") as f:
                for i in range(3):
                    l = random.choice(a)
                    r = random.choice(a)
                    f.write(name + "\t" + l +
                            "\t" + r + "\n")

    def _generate_mismatches_pairs(self):
        """
        Generate all mismatches pairs
        """
        for i, name in enumerate(os.listdir(self.data_dir)):
            if name == ".DS_Store":
                continue

            remaining = os.listdir(self.data_dir)
            remaining = [f_n for f_n in remaining if f_n != ".DS_Store"]
            # del remaining[i] # deletes the file from the list, so that it is not chosen again
            other_dir = random.choice(remaining)
            with open(self.pairs_filepath, "a") as f:
                for i in range(3):
                    file1 = random.choice(os.listdir(self.data_dir + name))
                    file2 = random.choice(os.listdir(self.data_dir +
                                                     other_dir))
                    f.write(name + "\t" + file1 + '\t' + other_dir + "\t" + file2 + '\n')
                f.write("\n")


if __name__ == '__main__':
    print('load all config')
    model_config = Map(yaml.safe_load(open('config/face_base.yaml')))
    cfg = _merge_a_into_b(model_config, get_cfg(), get_cfg(), [])
    data_dir = cfg.DATASETS.FOLDER + 'test/'
    pairs_filepath = cfg.DATASETS.FOLDER + "pairs.txt"
    img_ext = ".png"
    generatePairs = GeneratePairs(data_dir, pairs_filepath, img_ext)
    generatePairs.generate()