import matplotlib

matplotlib.use('agg')

import fnmatch
import os
import random

from keras.callbacks import ModelCheckpoint
from keras.metrics import binary_accuracy
from keras.optimizers import SGD
from keras.preprocessing.image import img_to_array, load_img, Iterator
import numpy as np

from res_model import get_model
from utils import binary_crossentropy_with_logits

input_shape = (160, 240)


class SegDirectoryIterator(Iterator):
    train_path = '/data/car_section/train/'
    mask_path = '/data/car_section/train_masks'

    def __init__(self):
        files = fnmatch.filter(
            os.listdir(self.train_path), '*.jpg')

        self.train_images = []
        self.validation_images = []

        for file in files:
            if random.random() > 0.1:
                self.train_images.append(file)
            else:
                self.validation_images.append(file)

        self.validation_data = np.zeros((len(self.validation_images), ) + input_shape + (3, ))
        self.validation_mask_data = np.zeros((len(self.validation_images), ) + input_shape + (1, ))

        count = 0
        for data_file in self.validation_images:
            img_x = load_img(os.path.join(self.train_path, data_file))
            img_y = load_img(os.path.join(
                self.mask_path, data_file.replace(".jpg", "_mask.gif")))

            img_x = img_x.resize((input_shape[1], input_shape[0]))
            img_y = img_y.resize((input_shape[1], input_shape[0]))

            x = img_to_array(img_x)
            y = img_to_array(img_y)
            y = y[:, :, :1]

            x = x / 255.
            y = y / 255.
            y = np.rint(y)
            self.validation_data[count] = x
            self.validation_mask_data[count] = y
            count += 1

        self.train_data = [None] * len(self.train_images)
        self.mask_data = [None] * len(self.train_images)
        super(SegDirectoryIterator, self).__init__(
            len(self.train_images), 10, True, 1)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(
                self.index_generator)

        # The transformation of images is not under thread lock so it can be
        # done in parallel
        batch_x = np.zeros((current_batch_size, ) + input_shape + (3, ))
        batch_y = np.zeros((current_batch_size, ) + input_shape + (1, ))

        for i, j in enumerate(index_array):
            if self.train_data[j] is not None:
                batch_x[i] = self.train_data[j]
                batch_y[i] = self.mask_data[j]
                continue
            data_file = self.train_images[j]
            img_x = load_img(os.path.join(self.train_path, data_file))
            img_y = load_img(os.path.join(
                self.mask_path, data_file.replace(".jpg", "_mask.gif")))

            img_x = img_x.resize((input_shape[1], input_shape[0]))
            img_y = img_y.resize((input_shape[1], input_shape[0]))

            x = img_to_array(img_x)
            y = img_to_array(img_y)
            y = y[:, :, :1]

            x = x / 255.
            y = y / 255.
            y = np.rint(y)

            self.train_data[j] = x
            self.mask_data[j] = y

            batch_x[i] = x
            batch_y[i] = y

        return (batch_x, batch_y)


filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


model = get_model(input_shape + (3, ))

# model.fit_generator(SegDirectoryIterator(), steps_per_epoch=1000, epochs=10)
sdi = SegDirectoryIterator()
model.fit_generator(sdi, steps_per_epoch=10, epochs=200, validation_data=[sdi.validation_data, sdi.validation_mask_data], callbacks=callbacks_list)
# model.fit(sdi.self.train_images_data, sdi.self.mask_images_data, batch_size=100, epochs=100)
