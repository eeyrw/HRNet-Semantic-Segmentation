import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from collections import namedtuple
import scipy.io
from tqdm import tqdm
import argparse

import _init_paths

from extraDs.cityscapes import Cityscapes
import models
from config import config
from config import update_config
import torchvision.transforms as transforms
import mytransforms as mytransforms
from constant import tusimple_row_anchor, culane_row_anchor
from dataset import LaneClsDataset, LaneTestDataset, LaneGenPseudoDataset
from datasetUtils import get_partial_dataset, split_dataset

input_mean = [0.485, 0.456, 0.406]
input_std = [0.229, 0.224, 0.225]

def getTusimpleLoader():
    target_transform = transforms.Compose([
        mytransforms.Scale((512, 1024)),
        mytransforms.MaskToTensor(),
    ])
    segment_transform = transforms.Compose([
        mytransforms.FreeScaleMask((36, 100)),
        mytransforms.MaskToTensor(),
    ])
    img_transform = transforms.Compose([
        transforms.CenterCrop((512, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(input_mean, input_std),
    ])

    train_dataset = LaneClsDataset('E:/Tusimple',
                                   os.path.join(
                                       'E:/Tusimple', 'train_gt.txt'),
                                   img_transform=img_transform, target_transform=target_transform,
                                   simu_transform=None,
                                   griding_num=100,
                                   row_anchor=tusimple_row_anchor,
                                   segment_transform=segment_transform, use_aux=False, num_lanes=4, load_name=True)

    sampler = torch.utils.data.SequentialSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, sampler=sampler, num_workers=4)

    return train_loader


def getCulaneLoader():
    target_transform = transforms.Compose([
        mytransforms.Scale((512, 1024)),
        mytransforms.MaskToTensor(),
    ])
    segment_transform = transforms.Compose([
        mytransforms.FreeScaleMask((36, 100)),
        mytransforms.MaskToTensor(),
    ])
    img_transform = transforms.Compose([
        transforms.CenterCrop((512, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(input_mean, input_std),
    ])

    train_dataset = LaneClsDataset('E:/CUlane',
                                   os.path.join(
                                       'E:/CUlane', 'list/train_gt.txt'),
                                   img_transform=img_transform, target_transform=target_transform,
                                   simu_transform=None,
                                   griding_num=200,
                                   row_anchor=culane_row_anchor,
                                   segment_transform=segment_transform, use_aux=False, num_lanes=4, load_name=True)

    sampler = torch.utils.data.SequentialSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=8, sampler=sampler, num_workers=4)

    return train_loader    

def resizeAndCropToTargetSize(img, width, height):
    rawW, rawH = img.size
    rawAspectRatio = rawW/rawH
    wantedAspectRatio = width/height
    if rawAspectRatio > wantedAspectRatio:
        scaleFactor = height/rawH
        widthBeforeCrop = int(rawW*scaleFactor)
        return img.resize((widthBeforeCrop, height), Image.BILINEAR). \
            crop(((widthBeforeCrop-width)//2, 0,
                  (widthBeforeCrop-width)//2+width, height))
    else:
        scaleFactor = width/rawW
        heightBeforeCrop = int(rawH*scaleFactor)
        return img.resize((width, heightBeforeCrop), Image.BILINEAR). \
            crop((0, (heightBeforeCrop-height)//2, width,
                  (heightBeforeCrop-height)//2+height))

def normalizeImage(imageTensor):
    maxVal = torch.max(imageTensor)
    minVal = torch.min(imageTensor)
    imageNormalized = (imageTensor-minVal)/(maxVal-minVal)
    return (imageNormalized*255).byte().cpu().numpy()

def tensorToCvBgr(image):
    # Step 1. chw to hwc Step 2. RGB to BGR
    img_bgr = normalizeImage(image).transpose((1, 2, 0))[..., ::-1]
    return img_bgr

def main():
    device = 'cuda'

    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args(
        args=["--cfg", "experiments\cityscapes\seg_hrnet_ocr_w48_demo.yaml"])
    update_config(config, args)

    # build model
    if torch.__version__.startswith('1'):
        module = eval('models.'+config.MODEL.NAME)
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    model = eval('models.'+config.MODEL.NAME +
                 '.get_seg_model')(config)

    model.to(device)


    weightFile = '../weight/hrnet_ocr_cs_trainval_8227_torch11.pth'
    # weightFile = 'pspnet_46_6.pth.tar'

    if os.path.isfile(weightFile):
        # print(("=> loading checkpoint '{}'".format('pspnet_46_6.pth.tar')))
        checkpoint = torch.load(weightFile)
        # torch.nn.Module.load_state_dict(model, checkpoint['state_dict'])
        compatible_state_dict = {}
        for k, v in checkpoint.items():
            if 'model.' in k:
                compatible_state_dict[k[6:]] = v
            elif 'loss.' in k:
                pass
            else:
                compatible_state_dict[k] = v

        model.load_state_dict(compatible_state_dict, strict=True)
        model.eval()

    cudnn.benchmark = True
    cudnn.fastest = True

    spp = models.spp.SPPLayer(4,pool_type='avg_pool')
    train_loader = getCulaneLoader()
    for imageFile in tqdm(train_loader):
        imgs = imageFile[0].cuda()

        with torch.no_grad():
            segOutput = model(imgs)[1]
            sppOut = spp(torch.sigmoid(segOutput))

        # imageBgrCV = cv2.cvtColor(np.asarray(resizeImage), cv2.COLOR_RGB2BGR)
        # imageBgrCV = np.zeros((512,1024,3),dtype=np.uint8)
        imageBgrCV = tensorToCvBgr(imgs[0])
        segOutput = segOutput[0]
        # segOutput: [class,h,w]
        # t = torch.sigmoid(segOutput)
        t = torch.argmax(segOutput, dim=0)
        segOutput = t.byte().cpu().numpy()
        # segOutput: [1,1,h,w]

        # colorMapMat = np.array([lb.color for lb in labels],dtype=np.uint8)[...,::-1] # RGB to BGR
        segImage = Cityscapes.decode_target(
            segOutput).astype(np.uint8)[..., ::-1]
        segImage = cv2.resize(segImage, (1024, 512),
                              interpolation=cv2.INTER_NEAREST)
        # segImage = colorMapMat[segOutput]
        imageBgrCV = cv2.addWeighted(imageBgrCV, 0.5, segImage, 0.5, 0)
        # imageBgrCV = segImage
        # imageBgrCV = cv2.resize(imageBgrCV,(3384//4,2710//4))

        cv2.imshow('L', imageBgrCV)
        # The following frees up resources and closes all windows
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
    cv2.destroyWindow('L')
    return


if __name__ == '__main__':
    main()
