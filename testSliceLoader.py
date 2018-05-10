from data.VolumeDataset import VolumeDataset
from data.slice_loader import SliceLoader
import torch
import time
import scipy.misc
from torchvision import transforms
import random

datadir = '/data/mri/data/raw-with-dti/train'
iProtocal = 'DTI-00'
oProtocal = 'T1'

volume_dataset = VolumeDataset(datadir, iProtocal, oProtocal, transform=True)
print(volume_dataset.get_slices_number())
loader = torch.utils.data.DataLoader(volume_dataset)
print("done initiating loader")
#sliceLoader = SliceLoader(volume_dataset, num_workers=2)
starttime = time.time()
cnt = 0
#for i, data in enumerate(sliceLoader):
#  #print(i)
#  cnt = cnt + 1
#print(time.time() - starttime)

def _toTensor(nibImg):
  img = scipy.misc.toimage(nibImg).convert('RGB')
  img = transforms.ToTensor()(img)
  img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
  return img

for e in range(1):
  sliceLoader = SliceLoader(volume_dataset, num_workers=2, isTest=True)
  cnt = 0
  for i, data in enumerate(sliceLoader):
    print(i)
    #print(data['A'].min(), data['A'].max(), data['B'].min(), data['B'].max())
    print(type(data['A']))
    print(data['A'].shape, data['B'].shape)
    imgA = _toTensor(data['A'][:,:,12])
    print(type(imgA), imgA.shape)
    if random.random() > 1.75 and random.random() < 0.8:
      print("save image")
      imgA = _toTensor(data['A'])
      imgB = _toTensor(data['B'])  
      imgA.save('{}_A.png'.format(cnt))
      imgB.save('{}_B.png'.format(cnt))
    cnt = cnt + 1
    if cnt > 1000:
      break
  print('finish epoch: {}'.format(e), cnt)
  print(time.time() - starttime)
#  #time.sleep(30)
