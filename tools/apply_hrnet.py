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
def main():
    # elif args.dataset == 'ApolloScape':
    num_class = 37  # merge the noise and ignore labels
    ignore_label = 255  # 0
    device = 'cuda'
    # model = models.PSPNet(num_class, base_model='resnet101',partial_bn=False).to(device)
    # model =  network.deeplabv3plus_mobilenet(num_classes=19, output_stride=16).to(device)

    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args(args=["--cfg","experiments\cityscapes\seg_hrnet_ocr_w48_demo.yaml"])
    update_config(config, args)

    # build model
    if torch.__version__.startswith('1'):
        module = eval('models.'+config.MODEL.NAME)
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    model = eval('models.'+config.MODEL.NAME +
                 '.get_seg_model')(config)
                 
    model.to(device)


    input_mean = [0.485, 0.456, 0.406]
    input_std = [0.229, 0.224, 0.225]



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


    img_transforms = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Lambda(lambda x:x*255),
        transforms.Normalize(input_mean, input_std),
    ])

    # image = Image.open(r"C:\Users\yuan\Desktop\highway45.mp4_snapshot_11.29_[2020.04.11_21.11.58].jpg")
    # resizeImage = resizeAndCropToTargetSize(image,3384//2,2710//2)
    # imageTensor = img_transforms(resizeImage)
    # imgs = torch.unsqueeze(imageTensor, 0).to(device)
    # img_w, img_h = image.size[0],image.size[1]

    # with torch.no_grad():
    #     segOutput = model(imgs).to(device)

    # imageBgrCV = cv2.cvtColor(np.asarray(resizeImage), cv2.COLOR_RGB2BGR)
    # segOutput=segOutput[0]
    # # segOutput: [class,h,w]
    # t = torch.sigmoid(segOutput)
    # t = torch.argmax(t,dim=0)
    # segOutput = t.byte().cpu().numpy()
    # # segOutput: [1,1,h,w]

    # colorMapMat = np.array([lb.color for lb in labels],dtype=np.uint8)[...,::-1] # RGB to BGR
    # segImage = colorMapMat[segOutput]
    # imageBgrCV = cv2.addWeighted(imageBgrCV, 1, segImage, 0.7, 0.4)
    # imageBgrCV = cv2.resize(imageBgrCV,(3384//4,2710//4))

    # while True:
    #     cv2.imshow('L', imageBgrCV)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # # The following frees up resources and closes all windows
    # cv2.destroyAllWindows()
    
    markClass = [
        ### Lane and road markings (4th channel) ###
        'background',
        'lane_solid_white',
        'lane_broken_white',
        'lane_double_white',
        'lane_solid_yellow',
        'lane_broken_yellow',
        'lane_double_yellow',
        'lane_broken_blue',
        'lane_slow',
        'stop_line',
        'arrow_left',
        'arrow_right',
        'arrow_go_straight',
        'arrow_u_turn',
        'speed_bump',
        'crossWalk',
        'safety_zone',
        'other_road_markings']

    vpClass = [
        ### Vanishing Points (5th channel) ###
        'background',
        'easy',
        'hard',
    ]

    # Get the list of all files in directory tree at given path
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(r'D:\VPGNet-DB-5ch'):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]

    colorMapMat = np.zeros((18, 3), dtype=np.uint8)
    laneColor = np.array([0, 255, 0], dtype=np.uint8)
    vpColorMapMat = np.array([[0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    for i in range(0, len(markClass)):
        if i != 0:
            colorMapMat[i] = np.random.randint(0, 255, dtype=np.uint8, size=3)

    for imageFile in tqdm(listOfFiles):
        data = scipy.io.loadmat(imageFile)
        rgb_seg_vp = data['rgb_seg_vp']
        rgb = rgb_seg_vp[:, :, 0: 3]
        img_bgr = rgb[:, :, :: -1]
        pilImage = Image.fromarray(np.uint8(rgb))
        resizeImage = resizeAndCropToTargetSize(pilImage,1024,512)
        imageTensor = img_transforms(resizeImage)
        imgs = torch.unsqueeze(imageTensor, 0).to(device)

        with torch.no_grad():
            segOutput = model(imgs)[1].to(device)

        imageBgrCV = cv2.cvtColor(np.asarray(resizeImage), cv2.COLOR_RGB2BGR)
        # imageBgrCV = np.zeros((512,1024,3),dtype=np.uint8)
        segOutput=segOutput[0]
        # segOutput: [class,h,w]
        # t = torch.sigmoid(segOutput)
        t = torch.argmax(segOutput,dim=0)
        segOutput = t.byte().cpu().numpy()
        # segOutput: [1,1,h,w]

        # colorMapMat = np.array([lb.color for lb in labels],dtype=np.uint8)[...,::-1] # RGB to BGR
        segImage = Cityscapes.decode_target(segOutput).astype(np.uint8)[...,::-1]
        segImage = cv2.resize(segImage,(1024,512),interpolation=cv2.INTER_NEAREST)
        # segImage = colorMapMat[segOutput]
        imageBgrCV = cv2.addWeighted(imageBgrCV, 0.5, segImage, 0.5, 0)
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
