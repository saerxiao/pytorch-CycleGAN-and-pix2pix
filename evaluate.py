import torch
import glob
import torch.nn as nn
import numpy as np
from models.vgg import Vgg16
import os
import re
from PIL import Image
import torchvision.transforms as transforms
import torch.autograd as Variable

cuda_id = 0
rootdir = "results"

mseLoss = nn.MSELoss()
vgg = Vgg16(requires_grad=False)
if cuda_id > -1:
  vgg = vgg.cuda(cuda_id)

def get_real_B(directory):
  l = glob.glob("{}/*_real_B.png".format(directory))
  return l

def compute_PSNR(img1, img2):
  img1 = img1 * 255
  img2 = img2 * 255
  mse = mseLoss(img1, img2)
  psnr = 20 * np.log10(255.) - 10 * np.log10(mse.data[0])
  return psnr

def compute_percept_loss(img1, img2):
  features1 = vgg(img1)
  features2 = vgg(img2)
  loss = mseLoss(features1.relu1_1, features2.relu1_1)
  return loss

def evaluate(name, which_epoch = "latest"):
  directory = "{}/{}/test_{}/images".format(rootdir, name, which_epoch)
  real_B_list  = get_real_B(directory)
  outputfile = "{}/{}/test_{}/eval.txt".format(rootdir, name, which_epoch)
  for real_B_path in real_B_list:
    m=re.match(r'(.*-\d+)_real_B.png', os.path.basename(real_B_path))
    sliceId = m.group(1)
    fake_B_path = "{}/{}_fake_B.png".format(directory, sliceId)
    real_B = Image.open(real_B_path).convert('RGB')
    fake_B = Image.open(fake_B_path).convert('RGB')  ## image is in [0,255]
    real_B = transforms.ToTensor()(real_B)  ## [0,1]
    fake_B = transforms.ToTensor()(fake_B)
    real_B = real_B.view(1, real_B.shape[0], real_B.shape[1], real_B.shape[2])
    fake_B = fake_B.view(1, fake_B.shape[0], fake_B.shape[1], fake_B.shape[2])
    real_B = torch.autograd.Variable(real_B)
    fake_B = torch.autograd.Variable(fake_B)
    if cuda_id > -1:
      real_B = real_B.cuda()
      fake_B = fake_B.cuda()
    psnr = compute_PSNR(real_B, fake_B)
    percept_loss = compute_percept_loss(real_B, fake_B)
    msg = "Id: {}, PSNR: {}, percept_loss: {}".format(sliceId, psnr, percept_loss.data[0])
    print(msg)
    with open(outputfile, "a") as f:
      f.write('%s\n' % msg)

evaluate("t123_mra_cgan") 
