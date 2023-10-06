import os
import shutil, errno
import pandas as pd
import random
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model



def resize_image(image, max_1d_res):
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


def pad_image(image, x_pix, y_pix):
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

def resize_data_set(input_dir, output_dir, max_dim):
    '''
    Resizes the images in the specified directory proportionally
    based on a specified maximum dimension
    Parameters
    ----------
    input_dir: string
        The input directory containing only image files to be resized
    output_dir: string
        The output directory
    max_dim: int
        The new max dimension of the image
    Returns
    -------
    No return value
    '''
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_num = 1 # initialize current file number
    file_count = len([name for name in os.listdir(input_dir) if (os.path.isfile(input_dir + '/' + name))])
    for file in os.listdir(input_dir):
        # check file type and if not an image move on
        input_file_path = input_dir + '/' + file
        if (os.path.isdir(input_file_path)):
            if file == 'resized':
                continue
            else:
                output_subdir = input_dir + '/' + 'resized' + '/' + file
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)
                for subfile in os.listdir(input_file_path):
                    # load the file
                    image = cv2.imread(input_dir + '/' + file + '/' + subfile)
 
                    image = resize_image(image, max_dim)
                    image = pad_image(image, max_dim, max_dim)
                    subfile_png = subfile[:-3] + 'png'
                    if not cv2.imwrite(os.path.join(output_subdir, subfile_png), image):
                        raise Exception("Could not write image")
                    else:
    
                        cv2.imwrite(os.path.join(output_subdir, subfile_png), image)
                    image = None # clear out the image
                    file_num+=1
        else:

            # load the file
            image = cv2.imread(input_dir + '/' + file)

            image = resize_image(image, max_dim)
            image = pad_image(image, max_dim, max_dim)
            file_png = file[:-3] + 'png'
            if not cv2.imwrite(os.path.join(output_dir, file_png), image):
                raise Exception("Could not write image")
            else:
                cv2.imwrite(os.path.join(output_dir, file_png), image)
            image = None # clear out the image
            file_num+=1


def evaluate_test(model, img_height, img_width, batch_size, data_dir):
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


def run_evaluation(keras_file, data_dir, img_width, img_height, batch_size):
        
    model_eval = load_model(keras_file)

    x_paths, pred, conf = evaluate_test(model=model_eval, img_height=img_height, img_width=img_width, batch_size=batch_size,data_dir=data_dir)
    loc_x_paths = []
    for x_path in x_paths:
        if (x_path.split('/')[-1][-3:] == 'png'):
            path = x_path.split('/')[-1][:-3] + 'bmp'
            loc_x_paths.append(path)
        else:
            path = x_path.split('/')[-1]
            loc_x_paths.append(path)           

    loc_x_paths = np.array(loc_x_paths)
    predictions = np.reshape(pred,(pred.shape[0],))
    d = {'file_path': loc_x_paths, 'prediction': predictions, 'confidence': conf}
    res = pd.DataFrame(data=d)
    res.to_csv("DOC_ADC_prediction_results.csv")
    
    # sort
    cls = pd.read_csv("DOC_ADC_prediction_results.csv")


def pipeline(file, data_dir, resize=True, move=False, train=False):
    img_height = 256
    img_width = 256
    batch_size=64
    if resize:
        rz_dir = data_dir + '/resized/'
        if not os.path.exists(rz_dir):
            os.makedirs(rz_dir)
        
        resize_data_set(input_dir=data_dir, \
               output_dir=rz_dir, \
               max_dim=img_height)
        
        if not train:
            # run evaluation based on keras model file and specified directory
            run_evaluation(file, rz_dir, img_width=img_width, img_height=img_height, batch_size=batch_size)
        
        shutil.rmtree(rz_dir)
                
    else:
        # run evaluation based on keras model file and specified directory
        run_evaluation(file, data_dir, img_width=img_width, img_height=img_height, batch_size=batch_size)