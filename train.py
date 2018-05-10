import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from data.VolumeDataset import VolumeDataset
from data.slice_loader import SliceLoader
import scipy.misc
from torchvision import transforms

opt = TrainOptions().parse()
if opt.dataset_mode == 'dti':
  is_dti = True

if is_dti:
  datadir = "{}/train".format(opt.dataroot)
  data_loader = VolumeDataset(datadir, opt.in_protocal, opt.out_protocal, transform=True)
  dataset_size = data_loader.get_slices_number()
else:
  data_loader = CreateDataLoader(opt)
  dataset = data_loader.load_data()
  dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)
total_steps = 0
errors_acc = {}

def _toTensor(nibImg):
  img = scipy.misc.toimage(nibImg).convert('RGB')
  img = transforms.ToTensor()(img)
  img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
  img = img.view(1,img.shape[0], img.shape[1], img.shape[2])
  return img

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    iter_data_time = time.time()
    epoch_iter = 0

    if is_dti:
      dataset = SliceLoader(data_loader, num_workers=2)
    for i, data in enumerate(dataset):
        if is_dti:
          data['A'] = _toTensor(data['A'])
          data['B'] = _toTensor(data['B'])
        iter_start_time = time.time()
        if total_steps % opt.print_freq == 0:
            t_data = iter_start_time - iter_data_time
        visualizer.reset()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data)
        model.optimize_parameters()

        if total_steps % opt.display_freq == 0:
            save_result = total_steps % opt.update_html_freq == 0
            if opt.dataset_mode == 'unaligned_array':
              visuals = model.get_current_numpy()
            else:
              visuals = model.get_current_visuals()
            visualizer.display_current_results(visuals, epoch, save_result)

        errors = model.get_current_errors()
        for k, v in errors.items():
            if k in errors_acc:
              errors_acc[k] = errors_acc[k] + v
            else:
              errors_acc[k] = v
        if total_steps % opt.print_freq == 0:
            errors = {}
            for k, v in errors_acc.items():
              errors[k] = v / opt.print_freq
            errors_acc = {}
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t, t_data)
            #if opt.display_id > 0:
                #visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

        iter_data_time = time.time()
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    model.update_learning_rate()
