import argparse
import os
import numpy as np
import torch
from torchvision import utils as tvutils
from data.VolumeDataset import VolumeDataset
from data.slice_loader import SliceLoader
import nibabel as nib
from torch.autograd import Variable
import scipy.misc
from torchvision import transforms
from options.test_options import TestOptions
from models.models import create_model
import pickle

opt = TestOptions().parse()
with open('scripts/test_'+ opt.name + '.pkl', 'wb') as f:
  pickle.dump(opt, f, pickle.HIGHEST_PROTOCOL)

eval_mode = False
model = create_model(opt)
if eval_mode:
  model.eval()
def _toTensor(nibImg):
  img = scipy.misc.toimage(nibImg).convert('RGB')
  img = transforms.ToTensor()(img)
  img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
  #img = img.view(1,img.shape[0], img.shape[1], img.shape[2])
  return img

def _RGBtoGray(A):
  tmp = A[:,0, ...] * 0.299 + A[:,1, ...] * 0.587 + A[:,2, ...] * 0.114
  #A = tmp.unsqueeze(1)
  return tmp

output_dir = web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
if not os.path.exists(output_dir):
  os.makedirs(output_dir)

cnt = 0
datadir = "{}/test".format(opt.dataroot)
volume_dataset = VolumeDataset(datadir, opt.in_protocal, opt.out_protocal, transform=True)
#dataLoader = SliceLoader(volume_dataset, num_workers=2, isTest=True)
for i, data in enumerate(volume_dataset):
  print((data['volId']))
  volInput = nib.Nifti1Image(data['A'], data['affine'])
  nib.save(volInput, "%s/%s-%s_transformed.nii.gz" % (output_dir, data['volId'], opt.in_protocal))
  nib.save(data['B_original'], "%s/%s-%s.nii.gz" % (output_dir, data['volId'], opt.out_protocal))

  model_input = torch.FloatTensor(data['A'].shape[2], 3, 128, 128)
  model_target = torch.FloatTensor(data['B'].shape[2], 3, 128, 128)
  for i in range(data['A'].shape[2]):
    model_input[i,:,:,:] = _toTensor(data['A'][:,:,i])
    model_target[i,:,:,:] = _toTensor(data['B'][:,:,i])
  data['A'] = model_input
  data['B'] = model_target
  model.set_input(data)
  model.test()
  
  output = model.fake_B.data.cpu()
  output = _RGBtoGray(output)
  
  outputImg = nib.Nifti1Image(output.permute(1,2,0).numpy(), data['affine'])
  nib.save(outputImg, "%s/%s-%s_transformed_predict.nii.gz" % (output_dir, data['volId'], opt.out_protocal)) 
  cnt = cnt + 1
  if cnt > 0:
    break

