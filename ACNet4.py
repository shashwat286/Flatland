import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np

# parameters for training
NUM_CHANNELS = 22
GRAD_CLIP = 90.0
RNN_SIZE = 1024
OBS_SIZE = 18 # Placeholder

def normalized_columns_initializer(std=1.0):
    pass

class ACNet:
    def __init__(self, scope, a_size, trainer, TRAINING, GLOBAL_NET_SCOPE, OBS_SIZE):
        # STUB
        pass

    def _build_net(self, inputs, TRAINING, a_size, RNN_SIZE, OBS_SIZE):
        # STUB
        pass
