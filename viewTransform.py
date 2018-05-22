import nibabel as nib
import scipy.misc
from util import html
import os
from nilearn.image import resample_img

rootdir = '/data/pix2pix-pytorch/pytorch-CycleGAN-and-pix2pix/results/dti00_t1_cgan/test_latest'
volId = 'IXI131-HH-1527'
web_dir = '{}/web'.format(rootdir)
img_dir = '{}/images'.format(web_dir)
if not os.path.exists(img_dir):
  os.makedirs(img_dir)

nSlices = 0
def save_images(data, typekey):
  nSlices = data.shape[2]
  for i in range(data.shape[2]):
    img = scipy.misc.toimage(data[:,:,i])
    img.save('{}/{}_{}_{}.png'.format(img_dir, volId, i, typekey))
  return nSlices


fname= '{}/{}-T1_transformed_predict.nii.gz'.format(rootdir, volId)
nSlices = save_images(nib.load(fname).get_data(), 'B_fake')

fname = '{}/{}-DTI-00_transformed.nii.gz'.format(rootdir, volId)
A_real_nii = nib.load(fname)
A_real_data = A_real_nii.get_data()
save_images(A_real_data, 'A_real')

fname = '{}/{}-T1.nii.gz'.format(rootdir, volId)
B_real_nii = nib.load(fname)
save_images(B_real_nii.get_data(), 'B_real')

data = resample_img(B_real_nii, A_real_nii.affine, target_shape=A_real_data.shape).get_data()
save_images(data, 'B_real_transformed')

webpage = html.HTML(web_dir, 'vol Id: %s' % volId)
for i in range(nSlices):
  webpage.add_header('slice [%d]' % i)
  ims, txts, links = [], [], []
  img_A_real = '{}_{}_A_real.png'.format(volId, i)
  img_B_real = '{}_{}_B_real.png'.format(volId, i)
  img_B_fake = '{}_{}_B_fake.png'.format(volId, i)
  img_B_real_t = '{}_{}_B_real_transformed.png'.format(volId, i)
  imgs = [img_A_real, img_B_fake, img_B_real_t, img_B_real]
  links = imgs
  txts = ['A_real', 'B_fake', 'B_real_t', 'B_real']
  webpage.add_images(imgs, txts, links, width=256)
webpage.save()


