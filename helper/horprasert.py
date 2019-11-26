import numpy as np
from skimage.filters import threshold_otsu


def BD(I, E, s):
    numer = np.sum((I * E) / np.power(s, 2), axis=2)
    denom = np.sum(np.power(E / s, 2))
    return numer / denom


def CD(I, E, s):
    alpha = BD(I, E, s)
    inner = np.power((I - (E * np.dstack([alpha] * 3))) / s, 2)
    #inner.shape
    return np.sqrt(np.sum(inner, axis=2))


def flatBD(flatI, E, s):            
    A = np.sum((flatI * E) / np.power(s, 2))
    B = np.sum(np.power(E / s, 2))
    return A / B

def flatCD(flatI, E, s):
    alpha = flatBD(flatI, E, s)
    inner = np.power((flatI - (E * alpha)) / s, 2)
    return np.sqrt(np.sum(inner))
    

def NCD(I, E, s, b):
    return CD(I, E, s) / b


def NBD(I, E, s, a):
    return (BD(I, E, s) - 1) / a


def horprasert_mask(I, E, s):
    """ Calculate mask using Horparsert bg removal code and Otsu thresholding.
    """
    CDI = CD(I, E, s)
    thresh = threshold_otsu(CDI)
    return CDI > thresh
    