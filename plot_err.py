import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_err(block, conv_func = float):
  if block:
    return conv_func(block[0].split(' ')[1])
  else:
    return None

def get_G_content(s):
  block = re.findall(r'G_content\: \d{1}.\d{3}', s)
  return get_err(block)

def get_G_GAN(s):
  block = re.findall(r'G_GAN\: \d{1}.\d{3}', s)
  return get_err(block)

def get_D_real(s):
  block = re.findall(r'D_real\: \d{1}.\d{3}', s)
  return get_err(block)

def get_D_fake(s):
  block = re.findall(r'D_fake\: \d{1}.\d{3}', s)
  return get_err(block)

def get_iters(s):
  block = re.findall(r'iters\: \d+', s)
  return get_err(block, conv_func=int)

def draw(name):
  filepath = "checkpoints/{}/loss_log.txt".format(name)
  x, y_G_content, y_G_GAN, y_D_real, y_D_fake = [], [], [], [], []
  with open(filepath, 'r') as f:
    content = f.readlines()
  
  for line in content:
    i = get_iters(line)
    if i:
      x.append(i)
      y_G_content.append(get_G_content(line))
      y_G_GAN.append(get_G_GAN(line))
      y_D_real.append(get_D_real(line))
      y_D_fake.append(get_D_fake(line))
  
  fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col')
  ax1.plot(x, y_G_content, 'bo')
  ax1.set_title('G_content')
  ax2.plot(x, y_G_GAN, 'bo')
  ax2.set_title('G_GAN')
  ax3.plot(x, y_D_real, 'bo')
  ax3.set_title('D_real')
  ax4.plot(x, y_D_fake, 'bo')
  ax4.set_title('D_fake')
  
  fig.suptitle(name)
  outputpath = "checkpoints/{}/err.png".format(name)
  fig.savefig(outputpath)

draw('t2_t1_L1Only')

