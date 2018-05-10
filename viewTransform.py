import nibabel as nib
import scipy.misc

fname = '/data/pix2pix-pytorch/pytorch-CycleGAN-and-pix2pix/results/dti00_t1_perceptOnly/test_latest/IXI131-HH-1527-T1_transformed_predict.nii.gz'
out = 'b_fake'
data = nib.load(fname).get_data()
for i in range(data.shape[2]):
  img = scipy.misc.toimage(data[:,:,i])
  img.save('{}_{}.png'.format(i, out))

