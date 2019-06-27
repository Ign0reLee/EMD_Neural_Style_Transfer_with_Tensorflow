import os, sys
import tensorflow as tf
import cv2
import gc
from tqdm import tqdm_notebook
import time
import Batch as batch
import EDM_Model as edm
import vgg.vgg19 as vgg


data = batch.Batch(Style_path="./Train_Example/Style", Content_path="./Train_Example/Content")

#Opt
learning_rate= 0.001
lambda_c = 1.
lambda_s = 5.
lambda_tv = 1e-5
mini_batch_size = 4
n_epoch = 1

test_check = 200
a_loss = 0
test_loss =0

#Tensor Flow GPU Memory
config = tf.ConfigProto()
config.gpu_options.allocator_type ='BFC'
config.gpu_options.allow_growth=True


tf.reset_default_graph()
train_data = enumerate(data._next(mini_batch_size, status='train'))
test_data = enumerate(data._next(mini_batch_size, status='test'))


with tf.Session(config=config) as sess:
    EDM = edm.Model(sess, lambda_c,lambda_s,lambda_tv, learning_rate)
    #EDM = Model(sess, lambda_c,lambda_s,lambda_tv, learning_rate)
    sess.run(tf.initialize_all_variables())
    sess.run(tf.local_variables_initializer())
    saver = tf.train.Saver(max_to_keep = 100)
    
    
    st_time = time.time()
    for i, batch in train_data:
        train_content, train_style = batch        
        loss, _ = EDM.train(train_style, train_content)
        
        a_loss += loss
        #print(np.shape(train_content), np.shape(train_style), "Loss : ", a_loss)
        del loss
        
        if i % test_check ==0:
            
            ed_time = time.time()
            print("Arrived Check Point..")
            print("Now Epoch : ", i, ", Time : ", ed_time - st_time,", Loss : ", a_loss/ (i +1))
            a_loss = 0
            
            print("Testing Start..")
            st_time = time.time()
            for a, test_batch in test_data:
                if a> 100: break
                test_content, test_style = test_batch
                loss = EDM.test(test_style, test_content)
                #print(np.shape(test_content), np.shape(test_style), "Loss : ", loss/(a+1))
                test_loss += loss
                
            ed_time = time.time()
            print("Testing Done.. Test Loss : ", test_loss/(a+1),", Time : ", ed_time - st_time)
            
            saver.save(sess, './Model/EDM_' + str(i) + '.ckpt')   
            st_time = time.time()
