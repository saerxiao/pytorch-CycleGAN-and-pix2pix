import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html
import ntpath
import numpy as np
import pickle

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
if opt.random_rotation:
  web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch), 'random_rotation')
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
npy_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch), 'numpy')
if not os.path.exists(npy_dir):
  os.makedirs(npy_dir)

# test
if opt.target_type == 'pdd':
  a, b = -1, 1
elif 'qform' in opt.target_type:
  with open('qform-minmax.pkl', 'rb') as f:
    rescale_range = pickle.load(f)
  if opt.target_type == 'qform':
    a025, b025 = rescale_range['min025'], rescale_range['max025']
    a134, b134 = rescale_range['min134'], rescale_range['max134']
  elif opt.target_type == 'qform025':
    a, b = rescale_range['min025'], rescale_range['max025']
  elif opt.target_type == 'qform134':
    a, b = rescale_range['min134'], rescale_range['max134']
  
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()
    img_path = model.get_image_paths()
    print('%04d: process image... %s' % (i, img_path))
    if i < opt.how_many_display:
      visualizer.save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio)
    
    # scale back result and save as npy
    data_np = model.get_current_numpy()
    if opt.target_type:
      short_path = ntpath.basename(img_path[0])
      name = os.path.splitext(short_path)[0]
      # scale back result and save as npy
      for label, im in data_np.items():
        #print(label, im.min(), im.max())
        output_name = '%s_%s' % (name, label)
        save_path = os.path.join(npy_dir, output_name)
        if label != 'real_A':
          if opt.target_type == 'pdd' or opt.target_type == 'qform025' or opt.target_type == 'qform134':
            im = im * (b - a) + a
          elif opt.target_type == 'qform':
            im_tosave = np.zeros(im.shape)
            im_tosave[:,:,0] = im[:,:,0] * (b025 - a025) + a025
            im_tosave[:,:,2] = im[:,:,2] * (b025 - a025) + a025
            im_tosave[:,:,5] = im[:,:,5] * (b025 - a025) + a025
            im_tosave[:,:,1] = im[:,:,1] * (b134 - a134) + a134
            im_tosave[:,:,3] = im[:,:,3] * (b134 - a134) + a134
            im_tosave[:,:,4] = im[:,:,4] * (b134 - a134) + a134
            im = im_tosave
        np.save(save_path, im)

webpage.save()
