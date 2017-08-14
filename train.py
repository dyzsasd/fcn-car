import matplotlib

matplotlib.use('agg')

from keras.metrics import binary_accuracy
from keras.optimizers import SGD
from keras.preprocessing.image import img_to_array, load_img

from res_model import get_model
from utils import binary_crossentropy_with_logits


model = get_model((1280, 1918, 3))
model.compile(
    loss=binary_crossentropy_with_logits,
    optimizer=SGD(lr=0.01 / 16, momentum=0.9),
    metrics=[binary_accuracy]
)

img_x = load_img('data/sample/fff9b3a5373f_16.jpg')
img_y = load_img('data/sample/fff9b3a5373f_16_mask.gif')

x = img_to_array(img_x)
x = x.reshape((1,) + x.shape)

y = img_to_array(img_y)
y = y.reshape((1,) + y.shape)
y = y[:, :, :, :1]

model.fit(x, y, batch_size=1, epochs=2)
