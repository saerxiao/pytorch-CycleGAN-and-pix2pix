import torch
import torch.utils.data as data
import nibabel as nib
import numpy as np
from nilearn.image import resample_img
from random import *
import os
import string
import scipy.misc
from torchvision import transforms
import re

def translate(shape, zoom):
  translation_range = 0.1 ## 0.1, 0.05, [-translation_range, translation_range]
  unit = np.multiply(shape, zoom)
  delta = np.multiply(np.random.rand(3)*translation_range*2-translation_range, unit)
  translate = np.array([[1,0,0,delta[0]],[0,1,0,delta[1]],[0,0,1,delta[2]],[0,0,0,1]])
  return translate

rot_range = 45. #45, 15, [-rot_range, rot_range]
def xRotation():
  theta = np.random.rand()*rot_range*2 - rot_range
  rotation = np.array([[1,0,0,0],
                        [0, np.cos(np.pi/180.*theta), -np.sin(np.pi/180.*theta), 0],
                        [0, np.sin(np.pi/180.*theta), np.cos(np.pi/180.*theta), 0],
                        [0,0,0,1]])
  return rotation

def yRotation():
  theta = np.random.rand()*rot_range*2 - rot_range
  rotation = np.array([[np.cos(np.pi/180.*theta), 0, np.sin(np.pi/180.*theta), 0],
                         [0, 1, 0, 0],
                         [-np.sin(np.pi/180.*theta), 0, np.cos(np.pi/180.*theta), 0],
                         [0, 0, 0, 1]])
  return rotation

def zRotation():
  theta = np.random.rand()*rot_range*2 - rot_range
  rotation = np.array([[np.cos(np.pi/180.*theta), -np.sin(np.pi/180.*theta), 0, 0],
                         [np.sin(np.pi/180.*theta), np.cos(np.pi/180.*theta), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
  return rotation

def scale():
  r = 0.2 #0.2, 0.1 [-0.2, 0.2]
  sx = 1 + np.random.rand()*r*2 - r
  sy = 1 + np.random.rand()*r*2 - r
  sz = 1 + np.random.rand()*r*2 - r
  scale = np.array([[sx, 0, 0, 0], [0, sy, 0, 0], [0, 0, sz, 0], [0, 0, 0, 1]])
  return scale

def togrey(img):
  imgsize = img.size()[1]
  img3c = img.new().resize_(3, imgsize, imgsize)
  img3c[0].copy_(img[0])
  img3c[1].copy_(img3c[0])
  img3c[2].copy_(img3c[0])
  return img3c

class VolumeDataset(data.Dataset):
  def __init__(self, rootdir, inputprotocal, outputprotocal, transform=None, nSlices=None):
    self.rootdir = rootdir
    self.transform = transform
  
    volumeIds = [] 
    inputvolumes = []
    outputvolumes = []
    for filename in os.listdir(rootdir):
      if 'T1' in filename:
        m = re.match(r'(.*)-T1-*', os.path.basename(filename))
        volumeIds.append(m.group(1))
      #index = string.find(filename, '-' + inputprotocal)
      #if index > 0:
      #  volumeIds.append(filename[:index])
        #inputvolumes.append(filename)
        #volId = filename[:index]
        #outputvolumes.append(volId + '-' + outputprotocal + '.nii.gz'

    if not nSlices:
      nSlices = 0
      for volId in volumeIds:
        inputfile = ''.join([rootdir, '/', volId, '-', inputprotocal, '.nii.gz'])
        inputdata = nib.load(inputfile).get_data()
        nSlices = nSlices + inputdata.shape[2]
    self.number_of_slices = nSlices

    self.volumeIds = volumeIds
    self.rootdir = rootdir
    self.inputprotocal = inputprotocal
    self.outputprotocal = outputprotocal
    self.transform = transform
    #self.inputvolumes = inputvolumes
    #self.outputvolumes = outputvolumes

  def __getitem__(self, idx):
    volId = self.volumeIds[idx]
    inputfile = ''.join([self.rootdir, '/', volId, '-', self.inputprotocal, '.nii.gz'])
    targetfile = ''.join([self.rootdir, '/', volId, '-', self.outputprotocal, '_resliced.nii.gz'])
    inputdata = nib.load(inputfile)
    target = nib.load(targetfile)
    inputimg = inputdata.get_data()
    targetimg = target.get_data()
    
    if self.transform:
      ## create affine matrix
      transform = translate(inputimg.shape, inputdata.header.get_zooms())
      transform = np.dot(transform, zRotation())
      transform = np.dot(transform, xRotation())
      transform = np.dot(transform, yRotation())
      transform = np.dot(transform, scale())
      #transform = np.dot(scale(), np.dot(transform, np.dot(xRotation(), np.dot(yRotation(), zRotation()))))
      affine = np.dot(transform, inputdata.affine)
      inputimg = resample_img(inputdata, affine, target_shape=inputimg.shape).get_data()
      targetimg = resample_img(target, affine, target_shape=inputimg.shape).get_data()
      
    return {'volId': volId, 'A': inputimg, 'B': targetimg, 'affine': affine, 'B_original': target, 'A_paths':None, 'B_paths':None}

  def __len__(self):
    return len(self.volumeIds)

  def get_slices_number(self):
    return self.number_of_slices
