import os
import cv2
import numpy as np
from PIL import Image
from PIL.Image import fromarray


def valid_mask_true():
    '''
    Return an array of true masks for valid images.
    '''
    masks = []
    images = os.listdir('data/valid_mask')
    for i in sorted(images):
        
        img = np.array(Image.open('data/valid_mask/'+i))
        masks.append(img)
    masks = np.array(masks)
    
    return masks



def resize(prediction_valid):
    prediction_valid = [fromarray(img, 'L') for img in prediction_valid*255]
    prediction_valid = np.array([np.array(img.resize((240,320))) for img in prediction_valid])
    return prediction_valid