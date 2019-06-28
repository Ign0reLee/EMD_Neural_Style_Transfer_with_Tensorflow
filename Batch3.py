import os, sys
import numpy as np
import cv2
from itertools import *
import random
import time



class Batch():
    
    def __init__(self, Style_path, Content_path):
        
        self.Style_path = Style_path
        self.Content_path = Content_path
        self._data_path_load()
        
        
    def _data_path_load(self):
        
        self.train_data = []
        self.test_data = []
        self.ctr_idx = 0
        self.cte_idx = 0
        self.str_idx = 0
        self.ste_idx = 0
        
        print("Data Load Start...")
        st_time = time.time()

        self.str_path =list(os.path.join(self.Style_path, 'train', x) for x in os.listdir(os.path.join(self.Style_path, 'train')))
        self.ctr_path = list(os.path.join(self.Content_path, 'train' ,x) for x in os.listdir(os.path.join(self.Content_path, 'train')))
        self.ste_path = list(os.path.join(self.Style_path, 'test',x) for x in os.listdir(os.path.join(self.Style_path, 'test')))
        self.cte_path = list(os.path.join(self.Content_path, 'test',x) for x in os.listdir(os.path.join(self.Content_path, 'test')))

        print("Data Load End..")
        ed_time = time.time()
        print("Load Time.. ", str(ed_time - st_time), "sec")
              
        
    def _next(self, batch_size, status="train"):

        if status == 'train':

            while True:
                if self.ctr_idx + batch_size > len(self.ctr_path): 
                    self.ctr_idx = 0
                    break
                if self.str_idx+batch_size> len(self.str_path):
                    self.str_idx = 0
                    self.ctr_idx += batch_size
                else:
                    yield self._image_load(self.ctr_path[self.ctr_idx:self.ctr_idx+batch_size], self.str_path[self.str_idx:self.str_idx+batch_size])
                    self.str_idx += batch_size
        else:
            
                while True:
                    if self.cte_idx + batch_size > len(self.cte_path): 
                        self.cte_idx = 0
                        break
                    if self.ste_idx+batch_size> len(self.ste_path):
                        self.ste_idx = 0
                        self.cte_idx += batch_size
                    else:
                        yield self._image_load(self.cte_path[self.cte_idx:self.cte_idx+batch_size], self.ste_path[self.ste_idx:self.ste_idx+batch_size])
                        self.ste_idx += batch_size

           
                
        
    def _image_load(self, ctr, img_list):
        
        content_img = []
        style_img = []
        
        for ct in ctr:
            for img_name in img_list:


                try:
                    sts = self._image_resize(img_name)
                    cts = self._image_resize(ct)
                    style_img.append(sts)
                    content_img.append(cts)


                except ValueError:
                    pass
        
        return content_img, style_img
        
    def _image_resize(self, img_name):
        
        im = cv2.imread(img_name)
        b,g,r =cv2.split(im)
        im = cv2.merge([r,g,b])
        h,w,_ = np.shape(im)

        if h > w:
            ratio = h/w
            im = cv2.resize(im, (int(512 * ratio), 512))
            y = random.randint(0, int(512*ratio) - 256)
            x = random.randint(0, 256)
            #print(np.shape(im), y, x)

        else:
            ratio = w/h
            im = cv2.resize(im, (512 , int(512* ratio)))
            y = random.randint(0, 256)
            x = random.randint(0, int(512*ratio) - 256)
            #print(np.shape(im), y, x)

            
        return cv2.resize(im[x:x+256,y:y+256,:],(224,224))

        
            
        
