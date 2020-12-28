# Use the 'File' menu above to 'Save' after pasting in your imports, data, and function definitions.
# Remember that we can't use numpy math function on the GPU...

import numpy as np
#from numpy import exp
from numba import cuda
from numba import vectorize
import math


greyscales_gpu = cuda.to_device(greyscales)
weights_gpu = cuda.to_device(weights)

normalized_gpu = cuda.device_array(shape=(n,), 
                               dtype=np.float32)
weighted_gpu = cuda.device_array(shape=(n,), 
                             dtype=np.float32)


@vectorize(['float32(float32)'],target='cuda')
def gpu_normalize(x):
    return x / 255

@vectorize(['float32(float32, float32)'],target='cuda')
def gpu_weigh(x, w):
    return x * w

@vectorize(['float32(float32)'],target='cuda')
def gpu_activate(x): 
    return (math.exp(x) - math.exp(-x) ) / ( math.exp(x) + math.exp(-x) )
