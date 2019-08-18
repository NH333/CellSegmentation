import numpy as np
def ToVector(img):
    size = img.shape

    v = np.reshape(img,(img.size,1),order="F")

    return v