import matplotlib

matplotlib.use('agg')

import fnmatch
import os

from keras.metrics import binary_accuracy
from keras.optimizers import SGD
from keras.preprocessing.image import img_to_array, load_img, Iterator
import numpy as np

from res_model import get_model
from utils import binary_crossentropy_with_logits


class SegDirectoryIterator(Iterator):
    train_path = '/data/car_section/train/'
    mask_path = '/data/car_section/train_masks'

    def __init__(self):
        self.train_images = fnmatch.filter(
            os.listdir(self.train_path), '*.jpg')
        super(SegDirectoryIterator, self).__init__(
            len(self.train_images), 10, True, 1)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(
                self.index_generator)

        # The transformation of images is not under thread lock so it can be
        # done in parallel
        batch_x = np.zeros((current_batch_size, ) + (160, 240, 3))
        batch_y = np.zeros((current_batch_size, ) + (160, 240, 1))

        for i, j in enumerate(index_array):
            data_file = self.train_images[j]
            img_x = load_img(os.path.join(self.train_path, data_file))
            img_y = load_img(os.path.join(
                self.mask_path, data_file.replace(".jpg", "_mask.gif")))

            img_x = img_x.resize((240, 160))
            img_y = img_y.resize((240, 160))

            x = img_to_array(img_x)
            y = img_to_array(img_y)
            y = y[:, :, :1]

            x = x / 255.
            y = y / 255.

            batch_x[i] = x
            batch_y[i] = y

        return (batch_x, batch_y)


model = get_model((160, 240, 3))
model.compile(
    loss=binary_crossentropy_with_logits,
    optimizer=SGD(lr=0.01 / 16, momentum=0.9),
    metrics=[binary_accuracy]
)

# model.fit_generator(SegDirectoryIterator(), steps_per_epoch=1000, epochs=10)
sdi = SegDirectoryIterator()
model.fit_generator(sdi, steps_per_epoch=100, epochs=10)
# model.fit(sdi.self.train_images_data, sdi.self.mask_images_data, batch_size=100, epochs=100)
