import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob
import io as sys_io
from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB
import cv2

#import inference_onx

# normalize the predicted SOD probability map
transform_=transforms.Compose([RescaleT(320), ToTensorLab(flag=0)])

import utils

def image_loader(image):
    """load image, returns cuda tensor"""
 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sample = {'imidx':np.array([0]), 'image':image, 'label':image}
    shape= image.shape
    sample = transform_(sample)   # Perform rescale and color  format conversion RGB to tensorlab
    image = sample['image']
    image  = torch.unsqueeze(image,0)

    return image,[shape[1],shape[0],3] #assumes that you're using GPU



def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn



def main(cv2_img, image_file=None, image_bytes=None):

    # --------- 1. get image path and name ---------
    model_name='u2netp'

    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name, model_name + '.pth')
    #model_dir = '/home/ekbana/oddocker/background_removal/u2netp.pth'
    # --------- 3. model define ---------
    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB---------")
        net = U2NET(3,1)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3,1)

    #print(net)
    device = torch.device('cpu')
    net.load_state_dict(torch.load(model_dir,map_location=device))

    net.eval()

    print(net)
 
    img_original = cv2_img.copy()
    original = cv2_img.copy()
   
    inputs_test, im_size = image_loader(img_original)
    bk = np.full(img_original.shape, 255, dtype=np.uint8)  # white bk, same size and type of image

    inputs_test = inputs_test.type(torch.FloatTensor)
    print("inputs_test size", inputs_test.shape)


    inputs_test = Variable(inputs_test)

    d1,d2,d3,d4,d5,d6,d7= net(inputs_test)
    # normalization
    pred = d1[:,0,:,:]
    pred = normPRED(pred)
    predict = pred.squeeze()
    predict_np = predict.cpu().data.numpy()
    im = Image.fromarray(predict_np*255).convert('RGB')
    img = im.resize((im_size[0],im_size[1]),resample=Image.BILINEAR)
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray,2, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #cv2.imwrite(image_file,mask)

    return mask


def latest():
    if not os.path.exists("video"):
        os.makedirs("video")

    codec_temp = cv2.VideoWriter_fourcc(*'mp4v')
    temp_video_file="video/out2.mp4"
    out = cv2.VideoWriter(temp_video_file, codec_temp, 3, (640, 480))


    # --------- 1. get image path and name ---------
    model_name='u2net'

    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name, model_name + '.pth')

    model_dir = "palm_saved_models/u2netnew_bce_itr_196000_train_0.124817_tar_0.007448.pth"
    # --------- 3. model define ---------
    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB---------")
        net = U2NET(3,1)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3,1)

    #print(net)
    device = torch.device('cpu')
    net.load_state_dict(torch.load(model_dir,map_location=device))
  
    net.eval()

    file = "video/v2.mp4"
    cap  = cv2.VideoCapture(file)
    ret, frame = cap.read()

    print("==", ret)

    while ret:
        try:
            ret, cv2_img = cap.read()
            img_original = cv2_img.copy()
            original = cv2_img.copy()
           
            inputs_test, im_size = image_loader(img_original)
            bk = np.full(img_original.shape, 255, dtype=np.uint8)  # white bk, same size and type of image

            inputs_test = inputs_test.type(torch.FloatTensor)
            print("inputs_test size", inputs_test.shape)

         
            inputs_test = Variable(inputs_test)

            d1,d2,d3,d4,d5,d6,d7= net(inputs_test)
            # normalization
            pred = d1[:,0,:,:]
            pred = normPRED(pred)
            predict = pred.squeeze()
            predict_np = predict.cpu().data.numpy()
            im = Image.fromarray(predict_np*255).convert('RGB')
            img = im.resize((im_size[0],im_size[1]),resample=Image.BILINEAR)
            img = np.array(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray,2, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            img =  cv2.hconcat([img_original, mask ])
            img = cv2.resize(img, (640, 480))
            out.write(img)
        except:
            print("Stop")

    out.release()



if __name__ == "__main__":
    latest()
