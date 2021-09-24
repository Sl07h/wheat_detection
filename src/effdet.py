import pandas as pd
import numpy as np
import cv2
import gc
import math
import os
from skimage import io

import torch
import torchvision
from torchvision import datasets, transforms

from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader


results = []
detection_threshold = 0.5
batch_size = 1
kernel_size = stride_size = 512

path_field_day      = 'data/Field2_3_2019/07_25/'
path_log_bboxes     = path_field_day + 'log/Field_2_3.effdet.{}.csv'.format(kernel_size)
path_to_weight      = 'weights/fold0-best-all-states.bin'
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

from effdet.config import get_efficientdet_config
from effdet import EfficientDet, DetBenchPredict
from effdet.efficientdet import HeadNet

def load_net(checkpoint_path):
    config = get_efficientdet_config('tf_efficientdet_d5')
    config.num_classes = 1
    config.image_size=[kernel_size, kernel_size]
    config.norm_kwargs = dict(eps=.001, momentum=.01)
    net = EfficientDet(config, pretrained_backbone=False)
    net.class_net = HeadNet(config, num_outputs=config.num_classes)
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    del checkpoint
    gc.collect()
    model = DetBenchPredict(net)
    model.eval()
    # return model.cuda()
    return model

# source: https://www.kaggle.com/shonenkov/inference-efficientdet
model = load_net(path_to_weight)


# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html


# https://discuss.pytorch.org/t/how-to-load-images-from-different-folders-in-the-same-batch/18942
class WheatTestDataset(Dataset):

    def __init__(self, dir):
        self.dir = dir
        self.image_names = os.listdir(self.dir)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        path = os.path.join(self.dir, image_name)
        image = io.imread(path)
        H, W, _ = image.shape
        pad_w = kernel_size - W % kernel_size
        pad_h = kernel_size - H % kernel_size
        if H % kernel_size == 0:
            pad_h=0
        if W % kernel_size == 0:
            pad_w=0
        new_image = np.zeros((H+pad_h, W+pad_w, 3))
        new_image[:-pad_h, :-pad_w, :] = image
        
        # print(image.shape)
        # print(new_image.shape)
        # print(pad_w, (W+pad_w) / kernel_size)
        # print(pad_h, (H+pad_h) / kernel_size)
        new_image = new_image.transpose((2, 0, 1))
        # img_tensor = ToTensor()(new_image)
        img_tensor = new_image.astype(np.float32)
        img_tensor /= 255
        img_tensor = torch.from_numpy(img_tensor).float().to(device)
        print(img_tensor.shape)
        return img_tensor, image_name



def format_prediction_string(boxes, scores):
    pred_strings = []
    for score, bbox in zip(scores, boxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(score, bbox[0], bbox[1], bbox[2], bbox[3]))

    return " ".join(pred_strings)


def fix_coordinates(list_i, list_bbox, list_prob, W_boxes):
    for i, bbox, prob in zip(list_i, list_bbox, list_prob):
        x = i % W_boxes * kernel_size
        y = i // W_boxes * kernel_size
        # print(x,y)
        for i in range(bbox.shape[0]):
            bbox[i][0] += x 
            bbox[i][1] += y
    return np.vstack(list_bbox), np.hstack(list_prob)


# torch.device('cpu')
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
# in_features = model.roi_heads.box_predictor.cls_score.in_features
# model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
# model.load_state_dict(torch.load(path_to_weight, map_location=device))
# model.eval()
model.to(device)

test_data_loader = DataLoader(
    # WheatTestDataset(path_field_day+'src', transform = transforms.ToTensor()),
    WheatTestDataset(path_field_day),
    batch_size=1,
    shuffle=False,
    drop_last=False
)




for batch_images, batch_image_names in test_data_loader:
    image = batch_images[0]
    image_name = batch_image_names[0]
    _, H, W = image.shape
    print('\nИзображение {}: {}x{}'.format(image_name,W,H))
    # x = torch.from_numpy(np.expand_dims(image, axis=0)).float().to(device)
    # x = batch_images.contiguous(memory_format=torch.channels_last)
    # x = batch_images.permute(0,3,1,2)
    x = batch_images
    # print(x)
    # print(x.shape)
    kc, kh, kw = 3, kernel_size, kernel_size
    dc, dh, dw = 3, stride_size, stride_size
    ## https://discuss.pytorch.org/t/patch-making-does-pytorch-have-anything-to-offer/33850/11
    patches_unfold = x.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
    unfold_shape = patches_unfold.size()
    patches_unfold = patches_unfold.contiguous().view(-1, kc, kh, kw)
    patches = patches_unfold
    pos_list = []
    boxes_list = []
    scores_list = []


    # print(patches.shape[0])
    for k in range(patches.shape[0]//batch_size):
        batch = patches[k*batch_size:(k+1)*batch_size]
        outputs = model(batch)
        for i, image in enumerate(batch):
            # преобразуем в numpy
            # boxes = outputs[i]['boxes'].data.cpu().numpy()
            # scores = outputs[i]['scores'].data.cpu().numpy()
            boxes = outputs[i].detach().cpu().numpy()[:,:4]    
            scores = outputs[i].detach().cpu().numpy()[:,4]

            # фильтруем совсем слабые результаты
            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            scores = scores[scores >= detection_threshold]
            # (x0,y0), (x1,y1) -> X,Y,W,H
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
            # сохраняем относительные bbox-ы и их вероятность
            boxes_list.append(boxes)
            scores_list.append(scores)
            pos_list.append(k*batch_size+i)

    boxes_list, scores_list = fix_coordinates(pos_list, boxes_list, scores_list, W // kernel_size)
    # print(boxes_list)
    # print(scores_list)
    # print(pos_list)


    result = {
        'image_id': batch_image_names[0],
        'PredictionString': format_prediction_string(boxes_list, scores_list)
    }
    results.append(result)


test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
test_df.to_csv(path_log_bboxes, index=False)