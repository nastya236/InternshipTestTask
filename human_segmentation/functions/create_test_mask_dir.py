import numpy as np
from PIL import Image
from PIL.Image import fromarray
import os


def make_dir(predictions, names):
    '''
    Create a directory data/test_mask.
    
    Parametrs
    ---------
    predictions: np.array
        arrays of masks
    names: list[str]
        name of images from data/test
        
    '''
    path = 'data/test'
    
    #проверяем, существует ли нужная директория
    #if os.path.exists(path):
        
     #   os.rmdir(path) # удаляем, если уже есть
   # os.mkdir(path) # создаём новую и помещаем в нее каждую маску
    for name, pred in zip(names, predictions):
        img = fromarray(pred[...,0], 'L')
        img = img.resize((240,320))
        img.save(path+'_mask/'+name[:-4]+'.png')