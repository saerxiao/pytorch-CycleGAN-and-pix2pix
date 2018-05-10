import re
import glob
import numpy as np
from PIL import Image
from sets import Set
from scipy.misc import imsave
import os
from collections import OrderedDict
import util.util as util
from util import html
from util.visualizer import Visualizer
from options.test_options import TestOptions

loadsize = 256

name = 't123_mra_cgan_percept'
which_epoch = 'test_75'
directory = "/data/pix2pix-pytorch/pytorch-CycleGAN-and-pix2pix/results/{}/{}".format(name, which_epoch)

allfiles = sorted(glob.glob("{}/images/*_real_B.png".format(directory)))

def get_subject_id(filepath):
  m = re.match(r'(.*)-\d+-*', os.path.basename(filepath))
  return m.group(1)

subjectSlicesMap = {}
for fpath in allfiles:
  subjectId = get_subject_id(fpath)
  if subjectId not in subjectSlicesMap:
    subjectSlicesMap[subjectId] = []
  slices = subjectSlicesMap[subjectId]
  slices.append(fpath) 

def get_MIP(subjectId, fake_B=False):
  slices = subjectSlicesMap[subjectId]
  volume = np.zeros((len(slices), loadsize, loadsize, 3)) 
  N = len(slices)
  for i in range(N):
    fpath = slices[i]
    if fake_B:
      fpath = fpath.replace('real_B', 'fake_B')
    img = Image.open(fpath).convert('RGB')
    volume[i] = np.array(img)

  mip_N = np.amax(volume, axis=0)
  mip_W = np.amax(volume, axis=1)
  mip_H = np.amax(volume, axis=2) #util.tensor2im(np.amax(volume, axis=2))
  return mip_N, mip_W, mip_H

web_dir = "{}/mip".format(directory)
#if not os.paths.exists(web_dir):
#  os.makedirs(web_dir)
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = test, Epoch = %s' % (name, which_epoch))
#image_dir = "{}/images".format(web_dir)
#if not os.paths.exists(image_dir):
#  os.makedirs(image_dir)

opt = TestOptions().parse()
visualizer = Visualizer(opt)
for subjectId in subjectSlicesMap:
  mip_N_real, mip_W_real, mip_H_real = get_MIP(subjectId)
  mip_N_fake, mip_W_fake, mip_H_fake = get_MIP(subjectId, fake_B=True)
  visuals = OrderedDict([('mip_N_fake', mip_N_fake), ('mip_N_real', mip_N_real), ('mip_W_fake', mip_W_fake), ('mip_W_real', mip_W_real), ('mip_H_fake', mip_H_fake), ('mip_H_real', mip_H_real)])
  print(subjectId)
  visualizer.save_images(webpage, visuals, name=subjectId)
webpage.save()  

