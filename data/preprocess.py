from pathlib import Path
from torchvision import transforms as trans
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import cv2
import bcolz
import pickle
import mxnet as mx
from tqdm import tqdm


def load_bin(path, rootdir, transform, image_size=[112, 112]):
    if not rootdir.exists():
        rootdir.mkdir()
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    data = bcolz.fill([len(bins), 3, image_size[0], image_size[1]],
                      dtype=np.float32,
                      rootdir=rootdir,
                      mode='w')
    for i in range(len(bins)):
        _bin = bins[i]
        img = mx.image.imdecode(_bin).asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = Image.fromarray(img.astype(np.uint8))
        data[i, ...] = transform(img)
        i += 1
        if i % 1000 == 0:
            print('loading bin', i)
    print(data.shape)
    np.save(str(rootdir) + '_list', np.array(issame_list))
    return data, issame_list


def load_mx_rec(rec_path):
    save_path = rec_path / 'imgs'
    if not save_path.exists():
        save_path.mkdir()
    imgrec = mx.recordio.MXIndexedRecordIO(str(rec_path / 'train.idx'),
                                           str(rec_path / 'train.rec'), 'r')
    img_info = imgrec.read_idx(0)
    header, _ = mx.recordio.unpack(img_info)
    max_idx = int(header.label[0])
    for idx in tqdm(range(1, max_idx)):
        img_info = imgrec.read_idx(idx)
        header, img = mx.recordio.unpack_img(img_info)
        label = int(header.label[0])
        img = Image.fromarray(img[:, :, ::-1])
        label_path = save_path / str(label)
        if not label_path.exists():
            label_path.mkdir()
        img.save(label_path / '{}.jpg'.format(idx), quality=95)


if __name__ == '__main__':
    rec_path = Path('/workspace/jiangby/project/datasets/faces_glintasia')
    load_mx_rec(rec_path)

    bin_files = ['agedb_30', 'cfp_fp', 'lfw', 'cfp_ff']

    test_transform = trans.Compose(
        [trans.ToTensor(),
         trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    for i in range(len(bin_files)):
        load_bin(rec_path / (bin_files[i] + '.bin'), rec_path / bin_files[i],
                 test_transform)
