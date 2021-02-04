from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms as trans
from torchvision.datasets import ImageFolder
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import bcolz
import os


def get_train_dataset(imgs_folder):
    train_transform = trans.Compose([
        trans.RandomHorizontalFlip(),
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    ds = ImageFolder(imgs_folder, train_transform)
    class_num = ds[-1][1] + 1
    return ds, class_num


def get_train_loader(cfg):
    ds, class_num = get_train_dataset(Path(cfg.DATASETS.FOLDER) / 'imgs')

    loader = DataLoader(ds,
                        batch_size=cfg.SOLVER.IMS_PER_BATCH,
                        shuffle=True,
                        pin_memory=False,
                        num_workers=cfg.DATALOADER.NUM_WORKERS)
    if cfg.MODEL.HEADS.NUM_CLASSES == 0:
        cfg.MODEL.HEADS.NUM_CLASSES = class_num
    return loader, class_num


def get_val_data(data_path):
    agedb_30, agedb_30_issame = get_val_pair(data_path, 'agedb_30')
    cfp_fp, cfp_fp_issame = get_val_pair(data_path, 'cfp_fp')
    lfw, lfw_issame = get_val_pair(data_path, 'lfw')
    return agedb_30, cfp_fp, lfw, agedb_30_issame, cfp_fp_issame, lfw_issame


def get_val_pair(path, name):
    carray = bcolz.carray(rootdir=path / name, mode='r')
    issame = np.load(path / '{}_list.npy'.format(name))
    return carray, issame


def get_paths(lfw_dir, pairs):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
            path0 = add_extension(
                os.path.join(lfw_dir, pair[0],
                             pair[1]))
            path1 = add_extension(
                os.path.join(lfw_dir, pair[0], pair[2]))
            issame = True
        elif len(pair) == 4:
            path0 = add_extension(
                os.path.join(lfw_dir, pair[0], pair[1]))
            path1 = add_extension(
                os.path.join(lfw_dir, pair[2], pair[3]))
            issame = False
        if os.path.exists(path0) and os.path.exists(
                path1):  # Only add the pair if both paths exist
            path_list += (path0, path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs > 0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)

    return path_list, issame_list


def add_extension(path):
    if os.path.exists(path):
        return path
    else:
        raise RuntimeError('No file "%s" with extension png or jpg.' % path)


def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)