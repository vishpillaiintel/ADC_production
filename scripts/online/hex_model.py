# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 10:45:00 2023

@author: Vishakh Pillai - ATTD Yield Prediction & Modeling
"""


# file processing, numpy, pandas imports
import os
import shutil, errno
import pandas as pd
import random
import sys
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# tensorflow imports
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

def resize(image, max_1d_res):
    '''
    Resize an image proportionally based on a 
    new maximum res value for the larger dimension
    
    Parameters
    ----------
    image: OpenCV image
        An OpenCV image (i.e. read with imread)
    max_1d_res: int
        The maximum resolution of any one dimension
        (the smaller dimension will scale proportionally)
        
    Returns
    -------
    OpenCV image
        The resized input image
    '''
    x = None
    y = None

    if image.shape[0] > image.shape[1]: # if y > x
        y = max_1d_res # set the larger dimension to be y
    else:
        x = max_1d_res # otherwise x

    width = x or int(image.shape[1] * (y/image.shape[0]))
    height = y or int(image.shape[0] * (x/image.shape[1]))
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return resized


def pad(image, x_pix, y_pix):
    '''
    Pad an image to the desired x and y dimensions
    Input image must be equal to or smaller in x and y than the specified dims
    
    Parameters
    ----------
    image: OpenCV image
        An OpenCV image (i.e. read with imread)
    x_pix: int
        The desired x-dimension in pixels
    y_pix: int
        The desired y-dimension in pixels

    Returns
    -------
    OpenCV image
        The padded input image
    '''
    # ensure that the target shape is
    # smaller than the original shape
    assert x_pix >= image.shape[1]  
    assert y_pix >= image.shape[0]
    x_padding = int((x_pix - image.shape[1])/2)
    y_padding = int((y_pix - image.shape[0])/2)


    padded_img = cv2.copyMakeBorder(image, y_padding, y_padding, x_padding, x_padding, cv2.BORDER_CONSTANT, value=0)
    x_diff = x_pix - padded_img.shape[1]
    y_diff = y_pix - padded_img.shape[0]

    # if there is a difference it is because the required 
    # padding isn't symmetric; we must add pixels as 
    # appropriate to reach correct resolution on only
    # on side of the image

    padded_img = cv2.copyMakeBorder(padded_img, y_diff, 0, x_diff, 0, cv2.BORDER_CONSTANT, value=0)

    return padded_img

def resize_images(input_full_paths, output_dir, max_dim=256):
    '''
    Resizes image in the specified directory proportionally
    based on a specified maximum dimension. Also, image will be converted 
    to PNG.

    Parameters
    ----------
    input_full_paths: list
        The input paths containing BMP image files to be resized
    output_dir: str
        The output directory
    max_dim: int
        The new max dimension of the image

    Returns
    -------
    No return value
    '''
    for full_path in input_full_paths:
        if ('OVERLAY' not in full_path) and (full_path[-3:].lower() == 'bmp'):
            # load the file
            image = cv2.imread(full_path)
            image = resize(image, max_dim)
            image = pad(image, max_dim, max_dim)
            file = os.path.basename(full_path)
            file_png = file[:-3] + 'png'
            if not cv2.imwrite(os.path.join(output_dir, file_png), image):
                raise Exception("Could not write image")
            else:
                cv2.imwrite(os.path.join(output_dir, file_png), image)
            image = None # clear out the image
            
class HEX_Model():

    def __init__(self, model_path, output_folder, image_folder):
        '''
        Initializes ADC model with which model to utilize, image parameters to use,
        and batch size. Also gives output directory and image resize directory.
       
        Parameters
        ----------
        model_path: str
            Path that specifies the model file to be loaded
        image_folder: str
            Input image folder with images to be classified

        Returns
        -------
        No return value
        '''

        self.img_height = 256
        self.img_width = 256
        self.batch_size = 1
        self.output_path = os.path.join(output_folder,'DOC_ADC_prediction_results.csv')
        self.image_folder = image_folder
        self.model_path = model_path

    def evaluate_test(self, model, img_height, img_width, batch_size, data_dir):
        '''
        Uses tensorflow.utils.image_dataset_from_directory() to load images. Crop
        to aspect ratio is set to False as images should be already resized. Shuffle
        is set to false in order for predictions & file paths to be aligned correctly for 
        final output. Labels are None as they need to be assigned unsupervised.
        
        Parameters
        ----------
        model: Keras file
            model loaded from file in models folder
        data_dir: str
            path of image folder (resized or not)
        img_width: int
            width of images to load from image folder
        img_height: int 
            height of images to load from image folder
        batch_size: int
            batch size to use when loading from image folder

        Returns
        -------
        test_ds.file_paths: numpy Array
            List of relative paths for images
        predictions: numpy Array
            0 = FM, 1 = OPEN, 2 = SHORT, 3 = TRANS_FM
        confidences: numpy Array
            0 to 1 value shows confidence in prediction given from
        tensorflow model. 
        '''

        IMG_SHAPE = (img_height,img_width)
        test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        crop_to_aspect_ratio=False,
        labels=None,
        shuffle=False)
        confidences = (model.predict(test_ds))
        predictions = np.argmax(confidences, axis=1)
        confidences = confidences.max(axis=1)
        return test_ds.file_paths, predictions, confidences


    def run_evaluation(self, keras_file, data_dir, img_width, img_height, batch_size):
        '''
        Loads model file from models folder given model choice, and then uses evaluate_test() 
        to load and evaluate images image folder. Then, outputs CSV files with prediction &
        confidences.

        Parameters
        ----------
        keras_file: str
            path of model file
        data_dir: str
            path of image folder (resized or not)
        img_width: int
            width of images to load from image folder
        img_height: int
            height of images to load from image folder
        batch_size: int
            batch size to use when loading from image folder

        Returns
        -------
        cls: pandas DataFrame
            CSV file with predictions & confidences
        '''    

        model_eval = load_model(keras_file)
        x_paths, pred, conf = self.evaluate_test(model=model_eval, img_height=img_height, \
                                            img_width=img_width, batch_size=batch_size,data_dir=data_dir)
        loc_x_paths = []
        for x_path in x_paths:
            x_path = x_path.replace('\\', '/')
            if (x_path.split('/')[-1][-3:] == 'png'):
                path = x_path.split('/')[-1][:-3] + 'bmp'
                loc_x_paths.append(path)
            else:
                path = x_path.split('/')[-1]
                loc_x_paths.append(path)           

        loc_x_paths = np.array(loc_x_paths)
        predictions = np.reshape(pred,(pred.shape[0],))
        prediction_map = {0: 'FM', 1: 'OPEN', 2: 'SHORT', 3: 'TRANS_FM'}
        predictions_mapped = [prediction_map[pred] for pred in predictions]
        d = {'MAU_IMAGE': loc_x_paths, 'ADC_CLASS': predictions_mapped, 'CONF': conf}
        res = pd.DataFrame(data=d)
        res.to_csv(self.output_path)
        
        cls = pd.read_csv(self.output_path)
        return cls


    def pipeline(self):
        '''
        Evaluates on an image folder and outputs a csv file with predictions and confidences.
        Parameters
        ----------
        resize - Boolean
            If specified true, then model will resize images before feeding into model. Default is true
        as this is how current models have been trained.

        Returns
        -------
        cls - pandas DataFrame
            CSV file with predictions & confidences for each image.

        '''

        img_height = self.img_height
        img_width = self.img_width
        batch_size = self.batch_size
        file = self.model_path
        data_dir = self.image_folder
        
        # run evaluation based on keras model file and specified directory
        cls = self.run_evaluation(keras_file=file, data_dir=data_dir, \
                                img_width=img_width, img_height=img_height, batch_size=batch_size)
        
        return cls
