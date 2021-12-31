import tensorflow as tf
from tensorflow.keras import *
import numpy as np

face_data = tf.data.Dataset.from_tensor_slices(np.load("data/data.npy"))
