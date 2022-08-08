import torch
import yaml
import cv2
import os
from PIL import Image
import transforms
from matplotlib import pyplot as plt
import numpy as np
from data import BSDS_500, NYUD, PASCAL_Context, PASCAL_VOC12
import model

import time

if __name__ == '__main__':
    # load configures
    file_id = open('./cfgs.yaml')
    cfgs = yaml.load(file_id)
    file_id.close()

    net = model.DRNet(cfgs['vgg16-5stage']).eval()
    net.load_state_dict(torch.load('./model.pth'))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # dataset = PASCAL_Context(root=cfgs['dataset'], flag='test', transform=trans)
    dataset = BSDS_500(root=cfgs['dataset'], flag='test', VOC=True, transform=trans)
    # dataset = PASCAL_VOC12(root=cfgs['dataset'], flag='test', transform=trans)
    # dataset = NYUD(root=cfgs['dataset'], flag='test', rgb=False, transform=trans)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    t_time = 0
    t_duration = 0
    name_list = dataset.gt_list
    length = dataset.length
    t_time = 0
    t_duration = 0
    for i, data in enumerate(dataloader):
        with torch.no_grad():
            images = data['images']
            width, height = data['images'].size()[2:]
            images2x = torch.nn.functional.interpolate(data['images'], scale_factor=2, mode='bilinear', align_corners=False)
            images_half = torch.nn.functional.interpolate(data['images'], scale_factor=0.5, mode='bilinear', align_corners=False)
            star_time = time.time()
            images = images.to(device)
            prediction = net(images).cpu().detach().numpy().squeeze()
            images2x = images2x.to(device)
            prediction2x = net(images2x).cpu().detach().numpy().squeeze()
            images_half = images_half.to(device)
            prediction_half = net(images_half).cpu().detach().numpy().squeeze()

            prediction2x = cv2.resize(prediction2x, (height, width), interpolation=cv2.INTER_CUBIC)
            prediction_half = cv2.resize(prediction_half, (height, width), interpolation=cv2.INTER_CUBIC)
            output = (prediction + prediction2x + prediction_half)/3
            duration = time.time() - star_time
            t_time += duration
            t_duration += 1/duration
            print('process %3d/%3d image.' % (i, length))
            # prop
            if not os.path.exists('./test/1X/'):
                os.makedirs('./test1/1X/')
            if not os.path.exists('./test/multi/'):
                os.makedirs('./test/multi/')
            cv2.imwrite('./test/1X/' + name_list[i] + '.png', prediction * 255)
            cv2.imwrite('./test/multi/' + name_list[i] + '.png', output * 255)

    print('avg_time: %.3f, avg_FPS:%.3f' % (t_time/length, t_duration/length))

