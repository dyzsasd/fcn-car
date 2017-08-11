from keras.preprocessing.image import ImageDataGenerator

from res_model import get_model
from utils import SegDataGenerator


path = "data/sample"

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

weight_decay = 1e-4

model = get_model((1918, 1280))
