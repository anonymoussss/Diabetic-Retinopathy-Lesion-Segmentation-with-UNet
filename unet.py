#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 10:01:27 2018

@author: shawn
"""

import tensorflow as tf
import tensorlayer as tl
import numpy as np
import BatchDatasetReader as BDR
import read_Data_list as RDL
import sys
import time
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc


#path variable
logs_dir = 'logs/'
data_dir = 'Data/'

#basic constant variable
IMG_SIZE = 512
num_of_classes = 2
print_freq = 10

#training constant variable
MAX_EPOCH = int(3000+1)
batch_size = 3
test_batchsize = 3
train_nbr = 54
test_nbr = 27
step_every_epoch = int(train_nbr/batch_size)
test_every_epoch = int(test_nbr/test_batchsize)
learning_rate = tf.Variable(1e-5, dtype=tf.float32)

#the parameters of aupr
range_threshold = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

#flags parameters
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")

class Unet:
    def __init__(self, img_rows = IMG_SIZE, img_cols = IMG_SIZE):
        self.img_rows = img_rows
        self.img_cols = img_cols
    def load_data_util(self):
        image_options = {'resize': True, 'resize_size': IMG_SIZE} #resize all your images
        train_records, valid_records = RDL.read_dataset(data_dir) #get read lists
        train_dataset_reader = BDR.BatchDatset(train_records, image_options)
        validation_dataset_reader = BDR.BatchDatset(valid_records, image_options)
        return train_dataset_reader,validation_dataset_reader
    def model(self, image, is_train=True, reuse=False):
        with tf.variable_scope("model", reuse=reuse):
            tl.layers.set_name_reuse(reuse)
            W_init = tf.contrib.layers.xavier_initializer()
            input_image = tl.layers.InputLayer(image, name='input_layer') #input image
        
            conv2d_1 = tl.layers.Conv2d(input_image, 64, (3, 3), (1, 1),
                                        act=tf.nn.relu, padding='SAME', W_init=W_init, name='conv_1')
            conv2d_2 = tl.layers.Conv2d(conv2d_1, 64, (3, 3), (1, 1),
                                        act=tf.nn.relu, padding='SAME', W_init=W_init, name='conv_2')
            pool_1 = tl.layers.MaxPool2d(conv2d_2, (2, 2), (2,2), 
                                         padding='SAME', name='maxpool_1')
            conv2d_3 = tl.layers.Conv2d(pool_1, 128, (3,3), (1,1), 
                                        act=tf.nn.relu, padding='SAME', W_init=W_init, name='conv_3')
            conv2d_4 = tl.layers.Conv2d(conv2d_3, 128, (3,3), (1,1),
                                        act=tf.nn.relu, padding='SAME', W_init=W_init, name='conv_4')
            pool_2 = tl.layers.MaxPool2d(conv2d_4, (2,2), (2,2),
                                         padding='SAME', name='maxpool_2')
            conv2d_5 = tl.layers.Conv2d(pool_2, 256, (3,3), (1,1), 
                                        act=tf.nn.relu, padding='SAME', W_init=W_init, name='conv_5')
            conv2d_6 = tl.layers.Conv2d(conv2d_5, 256, (3,3), (1,1), 
                                        act=tf.nn.relu, padding='SAME', W_init=W_init, name='conv_6')
            pool_3 = tl.layers.MaxPool2d(conv2d_6, (2,2), (2,2),
                                         padding='SAME', name='maxpool_3')
            conv2d_7 = tl.layers.Conv2d(pool_3, 512, (3,3), (1,1), 
                                        act=tf.nn.relu, padding='SAME', W_init=W_init, name='conv_7')
            conv2d_8 = tl.layers.Conv2d(conv2d_7, 512, (3,3), (1,1), 
                                        act=tf.nn.relu, padding='SAME', W_init=W_init, name='conv_8')
            dropout_1 = tl.layers.DropoutLayer(conv2d_8, keep= 0.5, is_fix=True, is_train=is_train, name = 'drop_1')
            pool_4 = tl.layers.MaxPool2d(dropout_1, (2,2), (2,2),
                                         padding='SAME', name='maxpool_4')
            conv2d_9 = tl.layers.Conv2d(pool_4, 1024, (3,3), (1,1), 
                                        act=tf.nn.relu, padding='SAME', W_init=W_init, name='conv_9')
            conv2d_10 = tl.layers.Conv2d(conv2d_9, 1024, (3,3), (1,1),
                                         act=tf.nn.relu, padding='SAME', W_init=W_init, name='conv_10')
            dropout_2 = tl.layers.DropoutLayer(conv2d_10, keep= 0.5, is_fix=True, is_train=is_train, name = 'drop_2')
            upsampling_1 = tl.layers.UpSampling2dLayer(dropout_2, (2,2), name='upsample2d_1')
            conv2d_11 = tl.layers.Conv2d(upsampling_1, 512, (3,3), (1,1),
                                         act=tf.nn.relu, padding='SAME', W_init=W_init, name='conv_11')
            concat_1 = tl.layers.ConcatLayer([dropout_1, conv2d_11], 3, name ='concat_1')
            conv2d_12 = tl.layers.Conv2d(concat_1, 512, (3,3), (1,1), 
                                         act=tf.nn.relu, padding='SAME', W_init=W_init, name='conv_12')
            conv2d_13 = tl.layers.Conv2d(conv2d_12, 512, (3,3), (1,1),
                                     act=tf.nn.relu, padding='SAME', W_init=W_init, name='conv_13')
            upsampling_2 = tl.layers.UpSampling2dLayer(conv2d_13, (2,2), name='upsample2d_2')
            conv2d_14 = tl.layers.Conv2d(upsampling_2, 256, (3,3), (1,1),
                                         act=tf.nn.relu, padding='SAME', W_init=W_init, name='conv_14')
            concat_2 = tl.layers.ConcatLayer([conv2d_14,conv2d_6], 3, name='concat_2')
            conv2d_15 = tl.layers.Conv2d(concat_2, 256, (3,3), (1,1), 
                                         act=tf.nn.relu, padding='SAME', W_init=W_init, name='conv_15')
            conv2d_16 = tl.layers.Conv2d(conv2d_15, 256, (3,3), (1,1),
                                         act=tf.nn.relu, padding='SAME', W_init=W_init, name='conv_16')
            upsampling_3 = tl.layers.UpSampling2dLayer(conv2d_16, (2,2), name='upsample2d_3')
            conv2d_17 = tl.layers.Conv2d(upsampling_3, 128, (3,3), (1,1),
                                         act=tf.nn.relu, padding='SAME', W_init=W_init, name='conv_17')
            concat_3 = tl.layers.ConcatLayer([conv2d_17,conv2d_4], 3, name='concat_3')
            conv2d_18 = tl.layers.Conv2d(concat_3, 128, (3,3), (1,1), 
                                         act=tf.nn.relu, padding='SAME', W_init=W_init, name='conv_18')
            conv2d_19 = tl.layers.Conv2d(conv2d_18, 128, (3,3), (1,1),
                                         act=tf.nn.relu, padding='SAME', W_init=W_init, name='conv_19')
            upsampling_4 = tl.layers.UpSampling2dLayer(conv2d_19, (2,2), name='upsample2d_4')
            conv2d_20 = tl.layers.Conv2d(upsampling_4, 64, (3,3), (1,1),
                                         act=tf.nn.relu, padding='SAME', W_init=W_init, name='conv_20')
            concat_4 = tl.layers.ConcatLayer([conv2d_20,conv2d_2], 3, name='concat_4')
            conv2d_21 = tl.layers.Conv2d(concat_4, 64, (3,3), (1,1),
                                         act=tf.nn.relu, padding='SAME', W_init=W_init, name='conv_21')
            conv2d_22 = tl.layers.Conv2d(conv2d_21, 32, (3,3), (1,1),
                                         act=tf.nn.relu, padding='SAME', W_init=W_init, name='conv_22')
            conv2d_23 = tl.layers.Conv2d(conv2d_22, num_of_classes, (3,3), (1,1),
                                         padding='SAME', W_init=W_init, name='conv_23')
            #maybe conv2d_23 should not be activation!
            y = conv2d_23.outputs #transfer tl object to logits tensor
            pred = tf.argmax(y, dimension=3, name="prediction")

        return pred,y,conv2d_23
    def loss(self, logits, annotation):
        loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=tf.squeeze(annotation, squeeze_dims=[3]),
                                                                        name="entropy")))
        
        L2 = 0
        for p in tl.layers.get_variables_with_name('/W', True, True):
            L2 += tf.contrib.layers.l2_regularizer(0.004)(p)
        cost = loss + L2
        return cost
    def train(self, loss):
        #If use tf.nn.sparse_softmax_cross_entropy_with_logits ,
        #maybe loss will be NAN,because without clip
        #annotation = tf.cast(annotation,dtype = tf.float32)
        #prob = tf.nn.softmax(logits)
        #loss = -tf.reduce_mean(annotation*tf.log(tf.clip_by_value(prob,1e-11,1.0)))
        optimizer = tf.train.AdamOptimizer(learning_rate)
        var_list = tf.trainable_variables()
        grads = optimizer.compute_gradients(loss, var_list=var_list)
        train_op = optimizer.apply_gradients(grads)
        return train_op
    
#AUPR score
def computeConfMatElements(thresholded_proba_map, ground_truth):
    P = np.count_nonzero(ground_truth)
    TP = np.count_nonzero(thresholded_proba_map*ground_truth)
    FP = np.count_nonzero(thresholded_proba_map - (thresholded_proba_map*ground_truth))    
                                 
    return P,TP,FP
    
def computeAUPR(proba_map, ground_truth, threshold_list):
    proba_map = proba_map.astype(np.float32)
    proba_map = proba_map.reshape(-1)
    ground_truth = ground_truth.reshape(-1)
    precision_list_treshold = []
    recall_list_treshold = []
    #loop over thresholds
    for threshold in threshold_list:
        #threshold the proba map
        thresholded_proba_map = np.zeros(np.shape(proba_map))
        thresholded_proba_map[proba_map >= threshold] = 1
        #print(np.shape(thresholded_proba_map)) #(400,640)
                   
        #compute P, TP, and FP for this threshold and this proba map
        P,TP,FP = computeConfMatElements(thresholded_proba_map, ground_truth)       
            
        #check that ground truth contains at least one positive
        if (P > 0 and (TP+FP) > 0) :
            precision= TP*1./(TP+FP)
            recall = TP*1./P
        else:
            precision = 1
            recall = 0

        #average sensitivity and FP over the proba map, for a given threshold
        precision_list_treshold.append(precision)
        recall_list_treshold.append(recall)    
    
    precision_list_treshold.append(1)
    recall_list_treshold.append(0) 
    return auc(recall_list_treshold, precision_list_treshold)

def main(argv=None):    
    myUnet = Unet()
    image = tf.placeholder(tf.int32,[None,IMG_SIZE,IMG_SIZE, 3], name='image') #input gray images
    annotation = tf.placeholder(tf.int32, shape=[None, IMG_SIZE, IMG_SIZE, 1], name="annotation")
    image = tf.cast(image,tf.float32)
    annotation = tf.cast(annotation,tf.int32)

    # define inferences
    train_pred, train_logits, train_tlnetwork = myUnet.model(image, is_train=True, reuse=False)
    train_positive_prob = tf.nn.softmax(train_logits)[:, :, :, 1]
    train_loss_op = myUnet.loss(train_logits, annotation)
    train_op = myUnet.train(train_loss_op)
    
    test_pred, test_logits, test_tlnetwork = myUnet.model(image, is_train=False, reuse=True)
    test_positive_prob = tf.nn.softmax(test_logits)[:, :, :, 1]
    test_loss_op= myUnet.loss(test_logits, annotation)
    
    lr_assign_op = tf.assign(learning_rate, learning_rate / 10) #learning_rate decay
    
    #only visualize the test images
    #first lighten the annotation images
    visual_annotation = tf.where(tf.equal(annotation,1), annotation+254, annotation)
    visual_pred = tf.expand_dims(tf.where(tf.equal(test_pred,1), test_pred+254, test_pred), dim=3)
    tf.summary.image("input_image", image, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(visual_annotation, tf.uint8), max_outputs=2)
    tf.summary.image("pred_annotation", tf.cast(visual_pred, tf.uint8), max_outputs=2)
    
    print("Setting up summary op...")
    test_summary_op = tf.summary.merge_all()
    
    if FLAGS.mode == 'train':
        train_dataset_reader,validation_dataset_reader = myUnet.load_data_util()
    sess = tf.Session()
    
    print("Setting up Saver...")
    saver = tf.train.Saver(max_to_keep=2)
    summary_writer = tf.summary.FileWriter(logs_dir, sess.graph)
    
    tl.layers.initialize_global_variables(sess)
    sess.run(tf.global_variables_initializer())
    
    ckpt = tf.train.get_checkpoint_state(logs_dir)#if model has been trained,restore it
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)   
        print("Model restored...")

    for epo in range(MAX_EPOCH):
        start_time = time.time()
        train_loss, test_loss, train_aupr, test_aupr, train_auc, test_auc= 0, 0, 0, 0, 0, 0
        
        for s in range(step_every_epoch):
            train_images, train_annotations = train_dataset_reader.next_batch(batch_size)
            feed_dict = {image: train_images, annotation: train_annotations}
            tra_positive_prob, train_err, _ = sess.run([train_positive_prob, train_loss_op,train_op], feed_dict=feed_dict)
            
            #compute auc score
            temp_train_annotations = np.reshape(train_annotations,-1)
            temp_tra_positive_prob = np.reshape(tra_positive_prob,-1)
            train_sauc = roc_auc_score(temp_train_annotations, temp_tra_positive_prob)
            #compute aupr
            train_saupr = computeAUPR(tra_positive_prob.reshape(-1),train_annotations.reshape(-1), range_threshold)
            
            
            train_loss += train_err
            train_auc += train_sauc
            train_aupr += train_saupr
            
        if epo + 1 == 1 or (epo + 1) % print_freq == 0:
            train_loss = train_loss/step_every_epoch
            train_auc = train_auc/step_every_epoch
            train_aupr = train_aupr/step_every_epoch
            #visualize the training loss
            print("%d epoches %d took %fs" % (print_freq, epo, time.time() - start_time))
            print("   train loss: %f" % train_loss)
            print("   train auc: %f" % train_auc)
            print("   train aupr: %f" % train_aupr)
            
            train_summary = tf.Summary(value=[
                    tf.Summary.Value(tag="train_loss", simple_value=train_loss), 
                    tf.Summary.Value(tag="train_auc", simple_value=train_auc),
                    tf.Summary.Value(tag="train_aupr", simple_value=train_aupr)
                ])
            summary_writer.add_summary(train_summary, epo)
            
            for test_s in range(test_every_epoch):
                #get validation data
                valid_images, valid_annotations = validation_dataset_reader.next_batch(test_batchsize)
                #visualize the validation loss
                feed_dict= {image:valid_images,annotation:valid_annotations}
                valid_positive_prob, validation_err = sess.run([test_positive_prob, test_loss_op], feed_dict= feed_dict)
                #compute auc score
                temp_valid_annotations = np.reshape(valid_annotations,-1)
                temp_valid_positive_prob = np.reshape(valid_positive_prob,-1)
                test_sauc = roc_auc_score(temp_valid_annotations, temp_valid_positive_prob)
                #compute test aupr
                test_saupr= computeAUPR(valid_positive_prob.reshape(-1), valid_annotations.reshape(-1), range_threshold)
                
                test_loss += validation_err
                test_auc += test_sauc
                test_aupr += test_saupr
            test_loss = test_loss/test_every_epoch
            test_auc = test_auc/test_every_epoch
            test_aupr = test_aupr/test_every_epoch
            test_summary = tf.Summary(value=[
                    tf.Summary.Value(tag="test_loss", simple_value=test_loss), 
                    tf.Summary.Value(tag="test_auc", simple_value=test_auc),
                    tf.Summary.Value(tag="test_aupr", simple_value=test_aupr)
                ])
            summary_writer.add_summary(test_summary, epo)
            
            #visualize the test result(only visualize the last batchsize of this epoch)
            feed_dict= {image:valid_images,annotation:valid_annotations}
            summary_str = sess.run(test_summary_op, feed_dict = feed_dict)
            summary_writer.add_summary(summary_str, epo)
            
            #tensorboard flush
            summary_writer.flush()
            sys.stdout.flush()
        if epo == 1500 or epo == 2000:   
            sess.run(lr_assign_op)
        if epo % 3000 == 0:
            saver.save(sess, logs_dir + "model.ckpt", epo)
            print('the %d epoch , the model has been saved successfully' %epo)
            sys.stdout.flush()
            
if __name__ == '__main__':
    tf.app.run()
