import os, sys
import tensorflow as tf
import numpy as np
import cv2
import gc
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import time
import Batch as batch
import vgg.vgg19 as vgg



class Model():
    
    def __init__(self, sess, lambda_c, lambda_s, lambda_tv , learning_rate):
        
        self.sess = sess
        self.vgg_st = vgg.Vgg19()
        self.vgg_ct = vgg.Vgg19()
        self.vgg_ot = vgg.Vgg19()
        self.lambda_c, self.lambda_s, self.lambda_tv = lambda_c, lambda_s, lambda_tv
        self.learning_rate = learning_rate
        self._build_net()
        
    def _build_net(self):
        
        print("Model Build Strat..")
        st_time = time.time()
        self.Style = tf.placeholder(tf.float32, [None, 224,224,3])
        self.Content = tf.placeholder(tf.float32,[None, 224,224,3])
        
        self.Style_mean, self.Style_std = self._Style_Encoder()
        self.F_con = self._Content_Encoder()
        self.match = self._Statistic_Matching(self.F_con, self.Style_mean, self.Style_std)
        self.Output = self._Decoder(self.match)
        L_c, L_s, L_tv = self._Loss()
        
        self.Loss = tf.add(tf.add(tf.multiply(self.lambda_c, L_c), tf.multiply(self.lambda_s, L_s)), tf.multiply(self.lambda_tv, L_tv))
        #self.Loss = tf.add(tf.add(tf.matmul(self.lambda_c, L_c), tf.multiply(self.lambda_s, L_s)), tf.multiply(self.lambda_tv, L_tv))
        self.Optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.Loss)
        print("Model Build End..")
        ed_time = time.time()
        print("Ending Time.. ", str(ed_time - st_time), "sec")
        
    def _Style_Encoder(self):
        
        with tf.variable_scope("ST_En1"):
            conv = self.en_conv_layer(self.Style, 5, 1, 64)
        with tf.variable_scope("ST_En2"):
            conv = self.en_conv_layer(conv, 3, 2, 128)
        with tf.variable_scope("ST_En3"):
            conv = self.en_conv_layer(conv, 3, 2, 256)
        with tf.variable_scope("ST_En4"):
            conv = self.en_conv_layer(conv, 3, 2, 256)

        with tf.variable_scope("ST_En5"):
            res = self.residual_block(conv, 3,256)
        with tf.variable_scope("ST_En6"):
            res = self.residual_block(res, 3,256)
        with tf.variable_scope("ST_En7"):
            res = self.residual_block(res, 3,256)
        with tf.variable_scope("ST_En8"):
            res = self.residual_block(res, 3,256)
        
        pool = self.Global_Pooling(res)
        fc = self.fc_layer(pool, 512)
        
        return tf.reduce_mean(fc, axis=1), tf.math.reduce_std(fc, axis=1)
        
    def _Content_Encoder(self):
        
        with tf.variable_scope("CT_En1"):
            conv = self.en_conv_layer(self.Content, 5, 1, 64)
        with tf.variable_scope("CT_En2"):    
            conv = self.en_conv_layer(conv, 3, 2, 128)
        with tf.variable_scope("CT_En3"):
            conv = self.en_conv_layer(conv, 3, 2, 256)

        with tf.variable_scope("CT_En4"):
            res = self.residual_block(conv, 3,256)
        with tf.variable_scope("CT_En5"):
            res = self.residual_block(res, 3,256)
        with tf.variable_scope("CT_En6"):
            res = self.residual_block(res, 3,256)
        with tf.variable_scope("CT_En7"):
            res = self.residual_block(res, 3,256)
        
        return res
    
    
    def _Statistic_Matching(self, F_con, Style_mean, Style_std):
        
        F_con_mean = tf.reduce_mean(F_con, axis=[1,2])
        F_con_std = tf.math.reduce_std(F_con, axis=[1,2])
        

        
        F_con_mean = tf.reshape(F_con_mean, [-1,1,1,F_con_mean.get_shape().as_list()[1]])
        F_con_std = tf.reshape(F_con_std, [-1,1,1,F_con_std.get_shape().as_list()[1]])
        Style_mean = tf.reshape(tf.tile(Style_mean, [256]), [-1,1,1,256])
        Style_std = tf.reshape(tf.tile(Style_std, [256]), [-1,1,1,256])

        return tf.add(tf.multiply(tf.divide(tf.subtract(F_con, F_con_mean), F_con_std), Style_std), Style_mean)
        
    def _Decoder(self, inputs_):
        with tf.variable_scope("DC1"):
            res = self.residual_block(inputs_, 3, 256)
        with tf.variable_scope("DC2"):
            res = self.residual_block(res, 3, 256)
        with tf.variable_scope("DC3"):
            res = self.residual_block(res, 3, 256)
        with tf.variable_scope("DC4"):
            res = self.residual_block(res, 3, 256)

        shape = tf.shape(res)
        upsample1 = self.up_samples(res, 112)
        
        with tf.variable_scope("DC4"):
            conv1 = self.de_conv_layer(upsample1, 3, 1, 128)
            
        shape = tf.shape(conv1)
        upsample2 = self.up_samples(conv1, 224)
        
        with tf.variable_scope("DC5"):
            conv2 = self.de_conv_layer(upsample2, 3,1,64)
        
        with tf.variable_scope("DC6"):
            conv3 = self.de_conv_layer(conv2, 5,1,3, False)
        
        return conv3
    
    def _Loss(self):
        
        with tf.name_scope("Style_vgg"):
            
            self.vgg_st.build(self.Style)
            
            st_relu1_2, st_relu2_2, st_relu3_3, st_relu4_3 = self.vgg_st.conv1_2, self.vgg_st.conv2_2, self.vgg_st.conv3_3, self.vgg_st.conv4_3
            
        with tf.name_scope("Content_vgg"):
            self.vgg_ct.build(self.Content)
            
            ct_relu4_1 = self.vgg_ct.conv4_1
            
        with tf.name_scope("Output_vgg"):
            
            self.vgg_ot.build(self.Output)
            out_relu1_2, out_relu2_2, out_relu3_3, out_relu4_1, out_relu4_3 = self.vgg_ot.conv1_2, self.vgg_ot.conv2_2, self.vgg_ot.conv3_3, self.vgg_ot.conv4_1, self.vgg_ot.conv4_3
       
        L_c =  self.build_content_loss(out_relu4_1, ct_relu4_1)
        
        L_s1 = self.build_style_loss(out_relu1_2,st_relu1_2)
        L_s2 = self.build_style_loss(out_relu2_2,st_relu2_2)  
        L_s3 = self.build_style_loss(out_relu3_3,st_relu3_3)
        L_s4 = self.build_style_loss(out_relu4_3, st_relu4_3)
        L_s = L_s1 + L_s2 + L_s3 + L_s4
        
        L_tv = self.build_total_variation_loss(self.Output)
        
        return L_c, L_s, L_tv
                          

    def build_content_loss(self, current, target):
        #return tf.reduce_mean(tf.square(tf.math.squared_difference(current, target)))
        return tf.reduce_mean(tf.math.squared_difference(current, target))
    
    def build_style_loss(self, current, target):
        
        current_mean, current_var = tf.nn.moments(current, axes=[2,3], keep_dims=True)
        current_std = tf.sqrt(current_var)
        
        target_mean, target_var = tf.nn.moments(target, axes=[2,3], keep_dims=True)
        target_std = tf.sqrt(target_var)
        
#         mean_loss = tf.reduce_sum(tf.square(tf.math.squared_difference(current_mean, target_mean)))
#         std_loss = tf.reduce_sum(tf.square(tf.math.squared_difference(current_std,target_std)))

        mean_loss = tf.reduce_sum(tf.math.squared_difference(current_mean, target_mean))
        std_loss = tf.reduce_sum(tf.math.squared_difference(current_std,target_std))
        
        n = tf.cast(tf.shape(current)[0], dtype=tf.float32)
        mean_loss /= n
        std_loss /= n
        
        return mean_loss + std_loss
        
    def build_total_variation_loss(self, target):
        
        return tf.reduce_sum(tf.image.total_variation(target))

    
    def en_conv_layer(self, inputs_, kernel_, strides_, outputs_):
        
        with tf.variable_scope("Encoder_Conv_Layer"):
            
            conv  = tf.layers.conv2d(inputs=inputs_, filters=outputs_, kernel_size = kernel_, strides = strides_,padding='same')
            leaky_relu = tf.nn.leaky_relu(conv, alpha= 0.2)
            return leaky_relu
    
    def de_conv_layer(self, inputs_, kernel_, strides_, outputs_,relu=True):
        
        with tf.variable_scope("Decoder_Conv_Layer"):
            
            conv = tf.layers.conv2d(inputs=inputs_, filters=outputs_, kernel_size = kernel_, strides = strides_,padding='same')
            if relu:
                conv = tf.nn.relu(conv)
            return conv
    
    def fc_layer(self, inputs_, outputs_):
        
        with tf.variable_scope("Encoder_Fully_Connected_Layer"):
            
            return tf.layers.dense(inputs = inputs_, units = outputs_)
        
    def residual_block(self, inputs_, kernel_, outputs_):
        
        with tf.variable_scope("Residual_blocks"):

            short = inputs_
            conv = tf.layers.conv2d(inputs=inputs_, filters=outputs_, kernel_size = kernel_, strides = 1,padding='same')
            conv = tf.contrib.layers.batch_norm(conv, updates_collections=None)
            conv = tf.nn.relu(conv)

            conv2 = tf.layers.conv2d(inputs=conv, filters=outputs_, kernel_size = kernel_, strides = 1,padding='same')
            conv2 = tf.contrib.layers.batch_norm(conv2, updates_collections=None)

            conv2 = tf.add(short, conv2)
            return tf.nn.relu(conv2)
        
    def up_samples(self, inputs_, up_size):
        

        return tf.image.resize_nearest_neighbor(inputs_,  (up_size,  up_size))
    
    def Global_Pooling(self, inputs_):
        
        return tf.reduce_mean(inputs_, axis=[1,2])
    
    def train(self, style_data, content_data):
        return self.sess.run([self.Loss, self.Optim], feed_dict={self.Style:style_data,
                                                                self.Content:content_data})
    
    def test(self, style_data, content_data):
        return self.sess.run(self.Loss, feed_dict={self.Style:style_data,
                                                  self.Content:content_data})
    
    def predict(self, style_data, content_data):
        return self.sess.run(self.Output, feed_dict={self.Style:style_data,
                                                    self.Content:content_data})
    
