import os
import cv2
import math
import pandas as pd
from PIL import Image
import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import torch
import torch.nn as nn

from torch.autograd import Variable
import torchvision.transforms as standard_transforms

#from datasets.QNRF.setting import cfg_data

# from misc.utils import *

import sys
sys.path.insert(1, './models')
from VGG16_LCM_REG import VGG16_LCM_REG
#%%

class CrowdCounter(nn.Module):
    def __init__(self, weights_path):
        super(CrowdCounter, self).__init__()
        self.CCN = VGG16_LCM_REG(False)
        self.load_state_dict(torch.load(weights_path, map_location='cuda:0'))
        self.cuda()
        self.eval()
        self.loss_sum_fn = nn.L1Loss().cuda()
        self.SumLoss = True
    
    def run(self, image_path):
        """TODO: Docstring for run.

        :image_path: TODO
        :returns: TODO

        """
        img = Image.open(image_path)
        if img.mode == 'L':
            img = img.convert('RGB')
        mean_std = [[0.413525998592, 0.378520160913, 0.371616870165],
 [0.284849464893, 0.277046442032, 0.281509846449]]
        img_transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(*mean_std)])

        img = img_transform(img)

        with torch.no_grad():
            img = Variable(img[None, :, :, :]).cuda()
            pred_map = self.CCN(img)
        
        pred_value = np.sum(pred_map.cpu().data.numpy()[0, 0, :, :])
        print("count is:", pred_value)
        
        ''' pred counting map '''
        den_frame = plt.gca()
        image = pred_map.cpu().data.numpy()[0, 0, :, :]
        plt.imshow(image)

        den_frame.axes.get_yaxis().set_visible(False)
        den_frame.axes.get_xaxis().set_visible(False)
        den_frame.spines['top'].set_visible(False)
        den_frame.spines['bottom'].set_visible(False)
        den_frame.spines['left'].set_visible(False)
        den_frame.spines['right'].set_visible(False)
        
        save_dir = "./images/results"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        plt.savefig(os.path.join(save_dir, 'result' + '_predmap_' + str(int(pred_value + 0.5)) + '.jpg'),
                    bbox_inches='tight', pad_inches=0, dpi=150)             

        ''' pred image '''
        text = "count:" + str(int(pred_value + 0.5))
        img_cv = cv2.imread(image_path)
        cv2.putText(img_cv, text, (10,30), cv2.FONT_HERSHEY_PLAIN, 2.0, (0,255,255), 2)
        cv2.imwrite(os.path.join(save_dir, 'result' + '__predcount__' + str(int(pred_value + 0.5)) + '.jpg'), img_cv)
        


                
        
#%%
if __name__ == '__main__':
    weights_path = './models/QNRF_mae_86.61_mse_152.19.pth'
    image_path = "./images/crowd6.jpg"
    # img_cv = cv2.imread(image_path)
    # while 0xFF & cv2.waitKey(1) != ord('q'):
    #     cv2.imshow("img", img_cv)

    #cv2.destroyAllWindows()
    cc = CrowdCounter(weights_path)
    cc.run(image_path)
   