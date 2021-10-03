import keras_vggface
from keras_vggface import VGGFace
import tensorflow.python.keras.engine




print(keras_vggface.__version__)

model = VGGFace(model='resnet50')

print(model.summary())


