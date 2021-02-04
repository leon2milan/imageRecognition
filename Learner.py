import os
import os.path as osp
from pathlib import Path
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from matplotlib import pyplot as plt
plt.switch_backend('agg')
from PIL import Image
from torchvision import transforms as trans
from baseline import Baseline
from data.datasets import get_val_data, get_train_loader, get_paths, read_pairs
from metric import scores
from utils import l2_norm, hflip_batch, gen_plot, get_time


class FACERecognition(object):
    def __init__(self, cfg, inference=False, threshold=0.5):
        self.device = torch.device(
            "cuda") if cfg.MODEL.DEVICE == 'cuda' else torch.device("cpu")

        if not inference:
            print('load training data')
            self.dataloader, class_num = get_train_loader(cfg)

            print('load testing data')
            if cfg.TEST.MODE == 'face':
                self.agedb_30, self.cfp_fp, self.lfw, self.agedb_30_issame, self.cfp_fp_issame, self.lfw_issame = get_val_data(
                    self.dataloader.dataset.root.parent)
            else:
                pairs = read_pairs(os.path.join(cfg.DATASETS.FOLDER, 'pairs.txt'))

                self.data, self.data_issame = get_paths(os.path.join(cfg.DATASETS.FOLDER, 'test'), pairs)
                
            print('load model')
            self.model = Baseline(cfg)
            self.model = self.model.to(self.device)
            self.load_state(cfg)
            if cfg.SOLVER.OPT == 'SGD':
                self.optimizer = optim.SGD(
                    [{
                        'params': self.model.parameters()
                    }],
                    lr=cfg.SOLVER.BASE_LR,
                    momentum=cfg.SOLVER.MOMENTUM,
                    weight_decay=cfg.SOLVER.WEIGHT_DECAY)
            else:
                self.optimizer = optim.Adam(
                    [{
                        'params': self.model.parameters()
                    }],
                    lr=cfg.SOLVER.BASE_LR,
                    weight_decay=cfg.SOLVER.WEIGHT_DECAY)

            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=cfg.SOLVER.MAX_EPOCH,
                eta_min=cfg.SOLVER.ETA_MIN_LR)
            checkpoints = cfg.CHECKPOINT.SAVE_DIR
            os.makedirs(checkpoints, exist_ok=True)

            self.best_score = 0.
            self.best_threshold = 0.
        else:
            self.device = torch.device(
                "cuda") if cfg.TEST.DEVICE == 'cuda' else torch.device("cpu")
            print('load model')
            self.model = Baseline(cfg)
            self.model = self.model.to(self.device)
            self.load_state(cfg)
            self.threshold = threshold
            self.test_transform = trans.Compose([
                trans.ToTensor(),
                trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

    def load_state(self, cfg):
        if cfg.CHECKPOINT.RESTORE:
            os.makedirs(cfg.CHECKPOINT.SAVE_DIR, exist_ok=True)
            weights_path = osp.join(cfg.CHECKPOINT.SAVE_DIR,
                                    cfg.CHECKPOINT.RESTORE_MODEL)
            self.model.load_state_dict(torch.load(weights_path,
                                                  map_location=self.device),
                                       strict=False)
            print('loaded model {}'.format(weights_path))

    def save_state(self, cfg, save_name):
        save_path = Path(cfg.CHECKPOINT.SAVE_DIR)
        torch.save(self.model.state_dict(), save_path / save_name)

    def evaluate(self, cfg, carray, issame, nrof_folds=5, tta=False):
        self.model.eval()
        idx = 0
        embeddings = np.zeros([len(carray), cfg.MODEL.HEADS.EMBEDDING_DIM])
        batch_size = cfg.SOLVER.IMS_PER_BATCH
        with torch.no_grad():
            while idx + batch_size <= len(carray):
                batch = torch.tensor(carray[idx:idx + batch_size])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.to(self.device)) + self.model(
                        fliped.to(self.device))
                    embeddings[idx:idx + batch_size] = l2_norm(emb_batch)
                else:
                    embeddings[idx:idx + batch_size] = self.model(
                        batch.to(self.device)).cpu()
                idx += batch_size
            if idx < len(carray):
                batch = torch.tensor(carray[idx:])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.to(self.device)) + self.model(
                        fliped.to(self.device))
                    embeddings[idx:] = l2_norm(emb_batch)
                else:
                    embeddings[idx:] = self.model(batch.to(self.device)).cpu()
        tpr, fpr, accuracy, best_thresholds = scores(embeddings, issame,
                                                     nrof_folds)
        buf = gen_plot(fpr, tpr)
        roc_curve = Image.open(buf)
        roc_curve_tensor = trans.ToTensor()(roc_curve)
        return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor

    def train(self, cfg):
        self.model.train()
        step = 0
        for e in range(cfg.SOLVER.MAX_EPOCH):
            for data, labels in tqdm(self.dataloader,
                                     desc=f"Epoch {e}/{cfg.SOLVER.MAX_EPOCH}",
                                     ascii=True,
                                     total=len(self.dataloader)):
                data = data.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                loss_dict = self.model(data, labels)
                losses = sum(loss_dict.values())
                losses.backward()
                self.optimizer.step()

                accuracy = 0.0
                if step % cfg.TEST.SHOW_PERIOD == 0:
                    print(
                        f"Epoch {e}/{cfg.SOLVER.MAX_EPOCH}, Step {step}, CE Loss: {loss_dict.get('loss_cls')}, Triplet Loss: {loss_dict.get('loss_triplet')}, Circle Loss: {loss_dict.get('loss_circle')}, Cos Loss: {loss_dict.get('loss_cosface')}"
                    )
                if step % cfg.TEST.EVAL_PERIOD == 0:
                    if cfg.TEST.MODE == 'face':
                        accuracy, best_threshold, roc_curve_tensor = self.evaluate(
                            cfg, self.agedb_30, self.agedb_30_issame)
                        print("dataset {}, acc {}, best_threshold {}".format(
                            'agedb_30',
                            accuracy,
                            best_threshold,
                        ))
                        accuracy, best_threshold, roc_curve_tensor = self.evaluate(
                            cfg, self.lfw, self.lfw_issame)
                        print("dataset {}, acc {}, best_threshold {}".format(
                            'lfw', accuracy, best_threshold))
                        if accuracy > self.best_score:
                            self.best_score = accuracy
                            self.best_threshold = best_threshold
                            self.save_state(
                                cfg,
                                'model_{}_best_accuracy:{:.3f}_step:{}.pth'.format(
                                    get_time(), accuracy, step))
                        accuracy, best_threshold, roc_curve_tensor = self.evaluate(
                            cfg, self.cfp_fp, self.cfp_fp_issame)
                        print("dataset {}, acc {}, best_threshold {}".format(
                            'cfp_fp', accuracy, best_threshold))
                    else:
                        accuracy, best_threshold, roc_curve_tensor = self.evaluate(
                            cfg, self.data, self.data_issame)
                        print("dataset {}, acc {}, best_threshold {}".format(
                            'test', accuracy, best_threshold))
                        if accuracy > self.best_score:
                            self.best_score = accuracy
                            self.best_threshold = best_threshold
                            self.save_state(
                                cfg,
                                'model_{}_best_accuracy:{:.3f}_step:{}.pth'.format(
                                    get_time(), accuracy, step))
                    self.model.train()
                if step % cfg.TEST.SAVE_PERIOD == 0:
                    self.save_state(
                        cfg, 'model_{}_accuracy:{:.3f}_step:{}.pth'.format(
                            get_time(), accuracy, step))

                step += 1

            self.save_state(
                cfg, 'model_{}_step:{}.pth'.format(
                    get_time(), step))

            self.scheduler.step()

    def infer(self, faces, target_embs, tta=False):
        '''
        faces : list of PIL Image
        target_embs : [n, 512] computed embeddings of faces in facebank
        names : recorded names of faces in facebank
        tta : test time augmentation (hfilp, that's all)
        '''
        embs = []
        for img in faces:
            if tta:
                mirror = trans.functional.hflip(img)
                emb = self.model(
                    self.test_transform(img).to(self.device).unsqueeze(0))
                emb_mirror = self.model(
                    self.test_transform(mirror).to(self.device).unsqueeze(0))
                embs.append(l2_norm(emb + emb_mirror))
            else:
                embs.append(
                    self.model(
                        self.test_transform(img).to(self.device).unsqueeze(0)))
        source_embs = torch.cat(embs)

        if isinstance(target_embs, list):
            tmp = []
            for img in target_embs:
                if tta:
                    mirror = trans.functional.hflip(img)
                    tmp = self.model(
                        self.test_transform(img).to(self.device).unsqueeze(0))
                    tmp_mirror = self.model(
                        self.test_transform(mirror).to(
                            self.device).unsqueeze(0))
                    tmp.append(l2_norm(tmp + tmp_mirror))
                else:
                    tmp.append(
                        self.model(
                            self.test_transform(img).to(
                                self.device).unsqueeze(0)))
            target_embs = torch.cat(tmp)

        diff = source_embs.unsqueeze(-1) - target_embs.transpose(
            1, 0).unsqueeze(0)
        dist = torch.sum(torch.pow(diff, 2), dim=1)
        minimum, min_idx = torch.min(dist, dim=1)
        min_idx[minimum > self.threshold] = -1  # if no match, set idx to -1
        return min_idx, minimum