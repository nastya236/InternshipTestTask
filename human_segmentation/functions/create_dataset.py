import os
import pandas as pd
import numpy as np
from PIL import Image
from lib import encode_rle

def create_dataset(path):
    '''Create dataset with names of images and code of their masks.

    Parameters
    ----------
    path: str
        Path to data with images.
    
    Return
    ----------
    data_df: DataFrame object (pandas.core.frame.DataFrame)
    
    '''
    masks_d = []
    masks = os.listdir(path+'_mask')
    for i in masks:
        img,mask = i[:-4]+'.jpg', np.array(Image.open(path+'_mask/'+i))
        masks_d.append([img,encode_rle(mask)])
    return pd.DataFrame(masks_d)



def from_image_to_array(data):
    '''
    
    PARAMETRS
    ----------
    data: pd.DataFrame
        Original data (test, valid images)
    model: model
    
    RETURN
    ---------
    names: str
    img: np.array
    '''
    
    arrays = []

    for i in data.sort_values(0).iloc[:,0].values:
        names.append(i)
        img = cv2.imread(path+i)
        img = cv2.resize(img, (256,256))
        arrays.append(img)
    img = np.array(arrays)/ 255.
    
    return names, img