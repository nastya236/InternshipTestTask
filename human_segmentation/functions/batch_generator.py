import cv2
import numpy as np
from lib import decode_rle


def generator(gen_df, batch_size):
    '''
    Create a batch generator.
    
    Parametrs
    -----------
    gen_df: DataFrame object
        Train data.
    batch_size: int
        Size of batch.
      
    Return
    ----------
    geherator
    
    '''
    
    while True:
        x_batch = []
        y_batch = []
        
        for i in range(batch_size):
            img_name, mask_rle = gen_df.sample(1).values[0]
            img = cv2.imread('data/train/{}'.format(img_name))
            mask = decode_rle(mask_rle)
            mask = cv2.resize(mask, (256, 256))
            img = cv2.resize(img, (256,256))
            
            
            x_batch += [img]
            y_batch += [mask]

        x_batch = np.array(x_batch) / 255.
        y_batch = np.array(y_batch)

        yield x_batch, np.expand_dims(y_batch, -1)
        
        
        
def generator_val(gen_df, batch_size):
    while True:
        x_batch = []
        y_batch = []
        
        for i in range(batch_size):
            img_name, mask_rle = gen_df.sample(1).values[0]
            img = cv2.imread('data/valid/{}'.format(img_name))
            mask = decode_rle(mask_rle)
            mask = cv2.resize(mask, (256, 256))
            img = cv2.resize(img, (256,256))
            
            
            x_batch += [img]
            y_batch += [mask]

        x_batch = np.array(x_batch) / 255.
        y_batch = np.array(y_batch)

        yield x_batch, np.expand_dims(y_batch, -1)