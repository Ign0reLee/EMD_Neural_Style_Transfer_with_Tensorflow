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
        
        print("Data Load Start...")
        st_time = time.time()

        self.style_train_path =(os.path.join(self.Style_path, 'train', x) for x in os.listdir(os.path.join(self.Style_path, 'train')))
        self.content_train_path = (os.path.join(self.Content_path, 'train' ,x) for x in os.listdir(os.path.join(self.Content_path, 'train')))
        self.style_test_path = (os.path.join(self.Style_path, 'test',x) for x in os.listdir(os.path.join(self.Style_path, 'test')))
        self.content_test_path = (os.path.join(self.Content_path, 'test',x) for x in os.listdir(os.path.join(self.Content_path, 'test')))

        print("Data Load End..")
        ed_time = time.time()
        print("Load Time.. ", str(ed_time - st_time), "sec")
              
        
    def _next(self, batch_size, status="train"):
        
        self.ctr_path = tee(self.content_train_path)
        self.cte_path = tee(self.content_test_path)

        now_ctr_path = []
        now_str_path = []
        now_ctrt_path = []
        now_strt_path = []
        
        
        if status == 'train':
            
            while True:
                now_ctr_path.clear()

                for i in range(batch_size):
                    try:
                        now_ctr_path.append(next(self.ctr_path[0]))
                    except StopIteration:
                        break
                        
                self.str_path = tee(self.style_train_path)

                for strs in self.str_path[0]:
                    now_str_path.append(strs)
                    
                    
                    if len(now_str_path) == batch_size:
                        yield self._image_load(now_ctr_path, now_str_path)
                        now_str_path.clear()
                        
        else:
            
            while True:
                now_ctrt_path.clear()

                for i in range(batch_size):
                    try:
                        now_ctrt_path.append(next(self.cte_path[0]))
                    except StopIteration:
                        break
                        
                self.ste_path = tee(self.style_train_path)

                for strs in self.ste_path[0]:
                    now_strt_path.append(strs)
                    
                    
                    if len(now_strt_path) == batch_size:
                        yield self._image_load(now_ctrt_path, now_strt_path)
                        now_strt_path.clear()
            
           
                
        
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

        
            
        
