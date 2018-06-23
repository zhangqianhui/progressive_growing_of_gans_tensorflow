from utils import CelebA, save_images, mkdir_p
import os
import numpy as np
from scipy.ndimage.interpolation import zoom

samples_path = './test'
mkdir_p(samples_path)

batch_size = 16
train_list = CelebA('/home/wangbin/data/celebA/').getNextBatch(0, batch_size)
realbatch_array = CelebA.getShapeForData(train_list, resize_w=32)

low_realbatch_array = zoom(realbatch_array, zoom=[1, 0.5, 0.5, 1], mode='nearest')
low_realbatch_array = zoom(low_realbatch_array, zoom=[1, 2, 2, 1], mode='nearest')

realbatch_array = np.clip(realbatch_array, -1, 1)

low_realbatch_array = np.clip(low_realbatch_array, -1, 1)

save_images(realbatch_array[0: batch_size], [2, batch_size/2],
                                '{}/{:02d}_real.png'.format(samples_path, 0))

save_images(low_realbatch_array[0: batch_size], [2, batch_size/2],
                                '{}/{:02d}_low_real.png'.format(samples_path, 0))