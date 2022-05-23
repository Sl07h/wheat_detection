import cv2
import copy
import glob
import os
import numpy as np
import pandas as pd
from time import gmtime, strftime

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from effdet.config import get_efficientdet_config
from effdet.efficientdet import EfficientDet, HeadNet
from effdet.bench import DetBenchTrain, DetBenchPredict

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler
from torchmetrics.detection.mean_ap import MeanAveragePrecision



class Experiment():
    # 'frcnn', df_train, df_val, df_test, 4,  2, 'cuda:0'
    def __init__(
        self,
        detection_model = 'frcnn',
        df_train   = 'gwhd_2021_train.csv',
        df_val     = 'gwhd_2021_val.csv',
        df_test    = 'gwhd_2021_test.csv',
        batch_size = 3,
        num_epochs = 5,
        device_str = 'cuda:0',
        augs_str   = 'flip',
    ) -> None:
        self.df_train = df_train
        self.df_val   = df_val
        self.df_test  = df_test
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = torch.device(device_str) if torch.cuda.is_available() else torch.device('cpu')
        self.augs_str = augs_str

        self.num_workers = int(os.cpu_count() / 4)
        self.detection_model = detection_model        
        mode_str = 'train_test_val' if df_val[10:14] == 'test' else 'train_val_test'
        config_str = f'{mode_str}.{detection_model}.{batch_size}.{num_epochs}.{augs_str}'
        self.path_to_logs_loss    = f'research/train/logs_loss/{config_str}.csv'
        self.path_to_logs_metrics = f'research/train/logs_metrics/{config_str}.csv'
        self.path_to_weight       = f'research/train/weights/{config_str}.pth'
        if os.path.exists(self.path_to_logs_loss):
            print(f'[+] эксперимент был выполнен ранее: {self.path_to_logs_loss}')
            return None


    def create_model(self):
        ''' создаём модель и отправляем её на устройство '''
        if self.detection_model == 'frcnn':
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        elif self.detection_model == 'effdet':
            config = get_efficientdet_config('tf_efficientdet_d5')
            net = EfficientDet(config, pretrained_backbone=False)
            config.num_classes = 1
            config.image_size = [512, 512]
            net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))
            self.model = DetBenchTrain(net, config)
        else:
            print('[-] неправильное имя модели детекции')
        if self.model:
            self.model.to(self.device)

    
    def prepare_data(self, images, targets):
        images_new  = [image.to(self.device) for image in images]
        targets_new = []
        for i in range(len(targets)):
            d = {}
            d['boxes'] = targets[i]
            d['labels'] = torch.ones((targets[i].shape[0]), dtype=torch.int64)
            targets_new.append(d)
        targets_new = [{k: v.to(self.device) for k, v in t.items()} for t in targets_new]
        if self.detection_model == 'effdet':
            images_new = torch.stack(images_new)
            targets_new = {
                'bbox': [b['boxes'].float() for b in targets_new],
                'cls': [l['labels'].float() for l in targets_new]
            }
        return images_new, targets_new

    
    def train_one_epoch(self):
        ''' обучаем одну эпоху
        return: loss_train, loss_val [float, float]
        '''
        loss_train = Averager()
        loss_val   = Averager()
        self.model.train()

        for i, (images, targets) in enumerate(self.train_dl):
            images, targets = self.prepare_data(images, targets)
            loss_dict = self.model(images, targets)
            # обновляем счётчик функции ошибки
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            loss_train.send(loss_value)
            # boilerplate-код
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()
            print(' '*80+f'\rtrain: {i+1} / {self.n_train}  loss: {loss_train.value}', end='\r')

        with torch.no_grad():
            for i, (images, targets) in enumerate(self.val_dl):
                images, targets = self.prepare_data(images, targets)
                loss_dict = self.model(images, targets)
                # обновляем счётчик функции ошибки
                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()
                loss_val.send(loss_value)
                print(' '*80+f'\rval: {i+1} / {self.n_val}  loss: {loss_val.value}', end='\r')
        
        return loss_train.value, loss_val.value


    def calc_metric(self, dataloader):
        ''' считаем mAP, mAP_50, mAP_75 для набора данных '''
        self.model.eval()
        metric = MeanAveragePrecision().to(self.device)
        n = len(dataloader)
        for i, (images, targets) in enumerate(dataloader):
            images, targets = self.prepare_data(images, targets)
            preds = self.model(images)
            metric.update(preds, targets)
            print(' '*80+f'\reval: {i+1} / {n}', end='\r')

        res = metric.compute()
        mAP    = res['map'].item() * 100
        mAP_50 = res['map_50'].item() * 100
        mAP_75 = res['map_50'].item() * 100
        return mAP, mAP_50, mAP_75


    def get_transforms(self):
        if self.augs_str == 'flip':
            self.train_transforms = A.Compose([
                A.Resize(512, 512),
                A.Flip(0.5),
                ToTensorV2(p=1.0)
            ], p=1.0, bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
            self.valid_transform = A.Compose([
                A.Resize(512, 512),
                ToTensorV2(p=1.0)
            ], p=1.0, bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
        elif self.augs_str == 'max_aug':
            self.train_transforms = A.Compose([
                A.RandomSizedCrop(min_max_height=(800, 800), height=1024, width=1024, p=0.5),
                A.OneOf([
                    A.HueSaturationValue(
                        hue_shift_limit=0.2,
                        sat_shift_limit= 0.2, 
                        val_shift_limit=0.2, p=0.9),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, 
                        contrast_limit=0.2, p=0.9), 
                ],p=0.9),
                A.ToGray(p=0.01),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Resize(height=512, width=512, p=1),
                A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
                ToTensorV2(p=1.0)
            ], p=1.0, bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
            self.valid_transform = A.Compose([
                A.Resize(512, 512),
                ToTensorV2(p=1.0)
            ], p=1.0, bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
        else:
            print('[-] неправильный режим аугментации')


    def create_dataloaders(self):
        ''' создаём 3 загрузчика датасетка: train_dl, val_dl, test_dl '''
        self.get_transforms()
        dataset_train = WheatDataset(self.df_train, self.train_transforms)
        dataset_val   = WheatDataset(self.df_val,   self.valid_transform)
        dataset_test  = WheatDataset(self.df_test,  self.valid_transform)
        self.train_dl = DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn
        )
        self.val_dl = DataLoader(
            dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn
        )
        self.test_dl = DataLoader(
            dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn
        )
        self.n_train = len(self.train_dl)
        self.n_val   = len(self.val_dl)
        self.n_test  = len(self.test_dl)


    def save_model(self):
        if self.detection_model == 'frcnn':
            torch.save(self.model.state_dict(), self.path_to_weight)
        elif self.detection_model == 'effdet':
            pass
        else:
            print('[-] неправильное имя модели - не могу сохранить')

    def perform(self):
        ''' проводим опыт. обучаем модель и смотрим метрики на лучшей модели '''
        self.create_model()
        self.create_dataloaders()

        df = pd.DataFrame(columns=['time_start', 'loss_train', 'loss_val'])
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        loss_min = 9000

        for epoch in range(self.num_epochs):
            time_start = strftime("%Y-%m-%d %H:%M:%S", gmtime())
            loss_train, loss_val = self.train_one_epoch()
            df = df.append({
                'time_start': time_start,
                'loss_train': loss_train,
                'loss_val':   loss_val,
            }, ignore_index=True)
            # если функция потерь упала более, чем на 2% и эпоха > 5
            if epoch >= 5 and loss_val < loss_min  and float(loss_min - loss_val) / loss_min > 0.02 :
                loss_min = loss_val
                self.save_model()
            if lr_scheduler is not None:
                lr_scheduler.step(loss_val)
            print(f'Epoch #{epoch+1} loss_train: {loss_train:.2f} loss_val: {loss_val:.2f}')
        df.to_csv(self.path_to_logs_loss)
        
        # with torch.no_grad():
        #     train_mAP, train_mAP_50, train_mAP_75 = self.calc_metric(self.train_dl)
        #     val_mAP, val_mAP_50, val_mAP_75       = self.calc_metric(self.val_dl)
        #     test_mAP, test_mAP_50, test_mAP_75    = self.calc_metric(self.test_dl)
        #     df = pd.DataFrame({
        #         'time': time_start,
        #         'train_mAP': train_mAP, 'train_mAP_50': train_mAP_50, 'train_mAP_75': train_mAP_75,
        #         'val_mAP': val_mAP,     'val_mAP_50': val_mAP_50,     'val_mAP_75': val_mAP_75,
        #         'test_mAP': test_mAP,   'test_mAP_50': test_mAP_50,   'test_mAP_75': test_mAP_75,
        #     })
        #     df.to_csv(self.path_to_logs_metrics)








class WheatDataset(Dataset):

    def __init__(self, df_name, transforms=None):
        super().__init__()
        df = pd.read_csv(f'data/gwhd_2021/{df_name}')
        # сохраняем в поля класса
        self.image_list = df['image_name'].values
        self.boxes_list = [self.decode_string(item) for item in df["BoxesString"]]
        self.domain_list = df['domain'].values
        self.transforms = transforms
        assert len(self.image_list) == len(self.boxes_list) == len(self.domain_list)

    def decode_string(self, boxes_string):
        """
        Small method to decode the BoxesString
        """
        if boxes_string == "no_box":
            return np.zeros((0, 4))
        else:
            try:
                boxes = np.array([np.array([int(i) for i in box.split(" ")])
                                  for box in boxes_string.split(";")])
                return boxes
            except:
                print(boxes_string)
                print("Submission is not well formatted. empty boxes will be returned")
                return np.zeros((0, 4))

    def __getitem__(self, index: int):
        # 1. текущая строка датафрейма
        image_name = self.image_list[index]
        bboxes = self.boxes_list[index]
        domain = self.domain_list[index]
        # 2. считываем изображение
        image = cv2.imread(f'data/gwhd_2021/{domain}/{image_name}', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        # 3. [проводим аугментацию], нормализуем
        if self.transforms:
            # Albumentations can transform images and boxes
            transformed = self.transforms(image=image, bboxes=bboxes, labels=["wheat_head"]*len(bboxes))
            image = transformed['image']
            bboxes = transformed['bboxes']
        if len(bboxes) > 0:
            bboxes = torch.stack([torch.tensor(item) for item in bboxes])
        else:
            bboxes = torch.zeros((0, 4))
        return image, bboxes

    def __len__(self) -> int:
        return len(self.image_list)


class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


def collate_fn(batch):
    """
    Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

    :param batch: an iterable of N sets from __getitem__()
    :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
    """
    images = list()
    targets = list()
    for i, t in batch:
        images.append(i)
        targets.append(t)
    images = torch.stack(images, dim=0)
    return images, targets




# 3 выборки
df_train = 'gwhd_2021_train.csv'
df_val   = 'gwhd_2021_val.csv'
df_test  = 'gwhd_2021_test.csv'

# 1. обучение моделей без аугментации
# Experiment('frcnn', df_train, df_val, df_test, 4, 20, 'cuda:0', 'flip').perform()
# Experiment('frcnn', df_train, df_val, df_test, 8, 20, 'cuda:0', 'flip').perform()
# Experiment('frcnn', df_train, df_test, df_val, 4, 20, 'cuda:0', 'flip').perform()
# Experiment('frcnn', df_train, df_test, df_val, 8, 20, 'cuda:0', 'flip').perform()

# 2. нормальная аугментация
# Experiment('frcnn', df_train, df_val, df_test, 4, 20, 'cuda:0', 'max_aug').perform()
Experiment('frcnn', df_train, df_val, df_test, 8, 20, 'cuda:0', 'max_aug').perform()
# Experiment('frcnn', df_train, df_test, df_val, 4, 20, 'cuda:0', 'max_aug').perform()
# Experiment('frcnn', df_train, df_test, df_val, 8, 20, 'cuda:0', 'max_aug').perform()

# 3. аугментация тестовой выборки


# 4. сборка пазла и хитрый загрузчик
