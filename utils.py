#####
#Copyright by Okyaz Eminaga, 2017-2018
#This is part includes frequently used functions.
#####

import os
import numpy as np
import cv2
import csv
from matplotlib import pyplot as plt
import math

from PIL import Image
import xml.etree.ElementTree as ET
from openslide import OpenSlide
import keras.backend as K
import random

from scipy.ndimage import gaussian_filter
from skimage import color, filters, morphology
from skimage.color import rgb2hsv, rgb2grey, rgb2hed
from skimage.exposure import rescale_intensity
from skimage.morphology import square
import skimage
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from keras import layers
from scipy.ndimage.interpolation import map_coordinates
#General functions
#Batch Normalization
def Conv2DBNSLU(x, filters, kernel_size=1, strides=1, padding='same', activation="relu", name=""):
    x = layers.Conv2D(
        filters,
        kernel_size = kernel_size,
        strides=strides,
        padding=padding,
        name=name,
        use_bias=False)(x)
    x = layers.BatchNormalization(scale=False)(x)
    x = layers.Activation(activation)(x)
    return x

def GetLastWeight(directory, prefix="weights", ends=".h5"):
    files = [os.path.splitext(os.path.basename(i))[0] for i in os.listdir(directory) if os.path.isfile(os.path.join(directory,i)) and prefix in i and i.endswith(ends)]
    if (len(files)==0):
        return None
    best_epoch = 1
    weight_epochs = []
    for fil in files:
        weight_epochs.append(int(fil.split("-")[1]))
    best_epoch = max(weight_epochs)
    print(best_epoch)
    #Construct the file name
    file_name_weight = directory + '/weights-%02d.h5' % (best_epoch)
    return file_name_weight

def CropLayers(input_shape=(64, 64, 3), cropping_size=(64, 64), off_set=(0, 0)):
    seq_seq_left_right_top_bottom_itms = []
    for x in range(0, input_shape[0], cropping_size[0]):
        for y in range(0, input_shape[1], cropping_size[1]):
            seq_top_bottom = (off_set[0] + x, (off_set[0] + x + cropping_size[0]))
            seq_left_right = (off_set[1] + y, (off_set[1] + y + cropping_size[1]))
            if seq_top_bottom[1] > input_shape[0]:
                diff = seq_top_bottom[1] - input_shape[0]
                seq_top_bottom = (seq_top_bottom[0] - diff, input_shape[0])
            if seq_left_right[1] > input_shape[1]:
                diff = seq_left_right[1] - input_shape[1]
                seq_left_right = (seq_left_right[0] - diff, input_shape[1])

            cropped_slide = (seq_top_bottom[0], seq_top_bottom[1], seq_left_right[0], seq_left_right[1])
            seq_seq_left_right_top_bottom_itms.append(cropped_slide)
    return seq_seq_left_right_top_bottom_itms

def get_number_to_cut(cropping_size, input_shape):
    coordinations = ((0, 0),
                    (int(round(cropping_size[0] / 2.)), 0),
                    (0, int(round(cropping_size[1] / 2.))),
                    (int(round(cropping_size[0] / 2.)), int(round(cropping_size[1] / 2.))),
                    (int(round(cropping_size[0] / 4.)), 0),
                    (0, int(round(cropping_size[1] / 4.))),
                    (int(round(cropping_size[0] / 4.)), int(round(cropping_size[1] / 4.))))
    # Generate image patches with different off set
    counter = 0
    for coord in coordinations:
        itm_list_coordination = CropLayers(input_shape=input_shape, cropping_size=cropping_size, off_set=coord)
        counter = counter + len(itm_list_coordination)
    return counter

def make_folders(file_prefix, file_address):
    if not file_prefix.startswith("/"):
        file_prefix = "./" + file_prefix
    folders = file_address.split("/")
    if folders[-1] == "":
        folders = folders[:-1]
    current_add = ""
    #print(folders)
    for folder in folders:
        for root, dirnames, filenames in os.walk(file_prefix + current_add):
            if folder not in dirnames:
                os.system("mkdir " + file_prefix + current_add + "/" + folder)
            break
        current_add += "/" + folder

def EnhanceColor(img, clipLimit=12, tileGridSize=(10,10)):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    img_c = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return img_c

def stainspace_to_2d_array(ihc_xyz, channel):
    rescale = rescale_intensity(ihc_xyz[:, :, channel], out_range=(0,1))
    stain_array = np.dstack((np.zeros_like(rescale), rescale, rescale))
    grey_array = rgb2grey(stain_array)
    return grey_array

def Convert_to_OD(rgb_255, normalized=False):
    rgb_dec = (rgb_255+1.) / 256.
    OD = -1 * np.log10(rgb_dec)
    '''
    if normalized:
        p_squared_sum = (OD[:,:,0] ** 2) + (OD[:,:,1] ** 2) + (OD[:,:,1] ** 2)
        OD[:,:,0] = OD[:,:,0] / (np.sqrt(p_squared_sum)+1e-7)
        OD[:,:,1] = OD[:,:,1] / (np.sqrt(p_squared_sum)+1e-7)
        OD[:,:,2] = OD[:,:,2] / (np.sqrt(p_squared_sum)+1e-7)
    '''
    return OD

def Convert_to_HSV(rgb_255, normalized=False):
    img = rgb2hsv(rgb_255)
    #rgb_dec = (img+1.) / 256.
    #print(np.min(rgb_dec))
    #OD = -1 * np.log10(img)
    '''
    if normalized:
        p_squared_sum = (OD[:,:,0] ** 2) + (OD[:,:,1] ** 2) + (OD[:,:,1] ** 2)
        OD[:,:,0] = OD[:,:,0] / (np.sqrt(p_squared_sum)+1e-7)
        OD[:,:,1] = OD[:,:,1] / (np.sqrt(p_squared_sum)+1e-7)
        OD[:,:,2] = OD[:,:,2] / (np.sqrt(p_squared_sum)+1e-7)
    '''
    #plt.imshow(OD)
    #plt.show()
    return img

def LN_normalize(arr):
    """
    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    arr = arr.astype('float')
    # Do not touch the alpha channel
    for i in range(3):
        minval = arr[...,i].min()
        maxval = arr[...,i].max()
        if minval != maxval:
            arr[...,i] -= minval
            arr[...,i] *= (255.0/(maxval-minval))
    return arr

def Convert_to_HRD(image):
    from skimage.color import rgb2hed
    ''''
    img_new = np.zeros(image.shape, dtype='float')
    rgb_from_hrd = np.array([[0.644, 0.710, 0.285],
                    [0.0326, 0.873, 0.487],
                    [0.270, 0.562, 0.781]])

    #conv_matrix
    hrd_from_rgb = linalg.inv(rgb_from_hrd)
    #Seperate stain
    ihc_hrd = separate_stains(image, hrd_from_rgb)

    #img_new = LN_normalize(ihc_hrd)
    #Stain space conversion
    DAB_Grey_Array = stainspace_to_2d_array(ihc_hrd, 2)
    Hema_Gray_Array = stainspace_to_2d_array(ihc_hrd, 0)
    GBIred_Gray_Array = stainspace_to_2d_array(ihc_hrd, 1)
    img_new = np.stack((Hema_Gray_Array, DAB_Grey_Array,GBIred_Gray_Array), axis=-1)
    '''
    ihc_hed = rgb2hed(image)
    #stainspace_to_2d_array()
    return ihc_hed

def stainspace_to_2d_array(ihc_xyz, channel):
    rescale = rescale_intensity(ihc_xyz[:, :, channel], out_range=(-1,1))
    stain_array = np.dstack((np.zeros_like(rescale), rescale, rescale))
    grey_array = rgb2grey(stain_array)
    return grey_array

def normalize_mapping(source_array, source_mean, target_mean):
    for x in source_array:
        x[...] = x/source_mean
        x[...] = x * target_mean
    return source_array

def plot_log(filename, show=True):
    # load data
    keys = []
    values = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if keys == []:
                for key, value in row.items():
                    keys.append(key)
                    values.append(float(value))
                continue

            for _, value in row.items():
                values.append(float(value))

        values = np.reshape(values, newshape=(-1, len(keys)))
        values[:,0] += 1

    fig = plt.figure(figsize=(4,6))
    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
    fig.add_subplot(211)
    for i, key in enumerate(keys):
        if key.find('loss') >= 0 and not key.find('val') >= 0:  # training loss
            plt.plot(values[:, 0], values[:, i], label=key)
    plt.legend()
    plt.title('Training loss')

    fig.add_subplot(212)
    for i, key in enumerate(keys):
        if key.find('acc') >= 0:  # acc
            plt.plot(values[:, 0], values[:, i], label=key)
    plt.legend()
    plt.title('Training and validation accuracy')

    # fig.savefig('result/log.png')
    if show:
        plt.show()

def combine_images(generated_images, height=None, width=None):
    num = generated_images.shape[0]
    if width is None and height is None:
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num)/width))
    elif width is not None and height is None:  # height not given
        height = int(math.ceil(float(num)/width))
    elif height is not None and width is None:  # width not given
        width = int(math.ceil(float(num)/height))

    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image

def random_scale_img(img, mask, xy_range, lock_xy=False):
    if random.random() > xy_range.chance:
        return img, mask

    if not isinstance(img, mask, list):
        img = [img]
        mask = [mask]

    import cv2
    scale_x = random.uniform(xy_range.x_min, xy_range.x_max)
    scale_y = random.uniform(xy_range.y_min, xy_range.y_max)
    if lock_xy:
        scale_y = scale_x

    org_height, org_width = img[0].shape[:2]
    xy_range.last_x = scale_x
    xy_range.last_y = scale_y

    res_img = []
    for img_inst in img:
        scaled_width = int(org_width * scale_x)
        scaled_height = int(org_height * scale_y)
        scaled_img = cv2.resize(img_inst, (scaled_width, scaled_height), interpolation=cv2.INTER_CUBIC)
        if scaled_width < org_width:
            extend_left = (org_width - scaled_width) / 2
            extend_right = org_width - extend_left - scaled_width
            scaled_img = cv2.copyMakeBorder(scaled_img, 0, 0, extend_left, extend_right, borderType=cv2.BORDER_CONSTANT)
            scaled_width = org_width

        if scaled_height < org_height:
            extend_top = (org_height - scaled_height) / 2
            extend_bottom = org_height - extend_top - scaled_height
            scaled_img = cv2.copyMakeBorder(scaled_img, extend_top, extend_bottom, 0, 0,  borderType=cv2.BORDER_CONSTANT)
            scaled_height = org_height

        start_x = (scaled_width - org_width) / 2
        start_y = (scaled_height - org_height) / 2
        tmp = scaled_img[start_y: start_y + org_height, start_x: start_x + org_width]
        res_img.append(tmp)

    res_mask = []
    for img_inst in mask:
        scaled_width = int(org_width * scale_x)
        scaled_height = int(org_height * scale_y)
        scaled_img = cv2.resize(
            img_inst, (scaled_width, scaled_height), interpolation=cv2.INTER_CUBIC)
        if scaled_width < org_width:
            extend_left = (org_width - scaled_width) / 2
            extend_right = org_width - extend_left - scaled_width
            scaled_img = cv2.copyMakeBorder(
                scaled_img, 0, 0, extend_left, extend_right, borderType=cv2.BORDER_CONSTANT)
            scaled_width = org_width

        if scaled_height < org_height:
            extend_top = (org_height - scaled_height) / 2
            extend_bottom = org_height - extend_top - scaled_height
            scaled_img = cv2.copyMakeBorder(
                scaled_img, extend_top, extend_bottom, 0, 0,  borderType=cv2.BORDER_CONSTANT)
            scaled_height = org_height

        start_x = (scaled_width - org_width) / 2
        start_y = (scaled_height - org_height) / 2
        tmp = scaled_img[start_y: start_y +
                         org_height, start_x: start_x + org_width]
        res_img.append(tmp)

    return res_img, res_mask
# Add others as needed
probs = {'keep': 0.1,'elastic': 0.9}
def apply_transform(image, mask, img_target_cols):
        #prob_value = np.random.uniform(0, 1)
        #if prob_value > probs['keep']:
            # You can add your own logic here.
        sigma = np.random.uniform(img_target_cols * 0.20, img_target_cols * 0.20)
        image = elastic_transform(image, img_target_cols, sigma)
        mask = elastic_transform(mask, img_target_cols, sigma)

        # Add other transforms here as needed. It will cycle through available transforms with give probs

        #mask = mask.astype('float32') / 255.
        return image, mask

def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    assert len(image.shape) == 3

    image_d = image.copy()
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape[0:2]

    for i in range(image.shape[2]):
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                            sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                            sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
        image_d[:,:,i] = map_coordinates(image[:,:,i], indices, order=1).reshape(shape)

    return image_d

def image_generator_flow(generator,
                         reconstruction=False,
                         verbose=False,
                         classify=False,
                         spare_category=False,
                         class_to_mask=False,
                         generate_weight=False,
                         reshape=False,
                         run_one_vs_all_mode=False,
                         nb_class=2,
                         target_size=(64,64),
                         batch_size=30):
    while 1:
        if (verbose == True):
            print("Loading the images...")
        x_batch_tmp, y_batch_tmp = generator.next()

        if (verbose == True):
            print("Generate the batch images...")

        if classify:
            y_batch = np.zeros((y_batch_tmp.shape[0], nb_class), dtype=K.floatx())
            for i, (y) in enumerate(y_batch_tmp):
                number_of_px = y.shape[0] * y.shape[1]
                np_flatted = y.flatten()
                y_indx = np.count_nonzero(np_flatted)
                batch_yy_class = 0
                y_per = y_indx / number_of_px

                if (nb_class == 2):
                    if (y_indx > 0):
                        batch_yy_class = 1
                # print(batch_yy_class)
                y_batch[i, batch_yy_class] = 1.
        y_batches = []
        if spare_category:
            pass

        if class_to_mask:
            y_batches = np.zeros((y_batch_tmp.shape[0], target_size[0], target_size[1], nb_class), dtype=K.floatX())
            for i, (y) in enumerate(y_batch_tmp):
                y_batch_itm = np.zeros(target_size[0], target_size[1], nb_class)
                y_batch_itm[:,:,y] = 1.
                y_batches[i] = y_batch_itm

        if run_one_vs_all_mode:
            # Background
            y_batch_0 = np.zeros((y_batch_tmp.shape[0], y_batch_tmp.shape[1], y_batch_tmp.shape[2], 2),
                                     dtype=K.floatx())
            y_batch_0[:, :, :, 0] = np.add(y_batch_tmp[:, :, :, 1], y_batch_tmp[:, :, :, 2])
            y_batch_0[:, :, :, 0] = y_batch_0[:, :, :, 0] > 0
            y_batch_0[:, :, :, 1] = y_batch_tmp[:, :, :, 0]
            y_batches.append(y_batch_0)
            # Nucleous
            y_batch_1 = np.zeros((y_batch_tmp.shape[0], y_batch_tmp.shape[1], y_batch_tmp.shape[2], 2),dtype=K.floatx())
            y_batch_1[:, :, :, 0] = np.add(y_batch_tmp[:, :, :, 0], y_batch_tmp[:, :, :, 2])
            y_batch_1[:, :, :, 0] = y_batch_0[:, :, :, 0] > 0
            y_batch_1[:, :, :, 1] = y_batch_tmp[:, :, :, 1]
            y_batches.append(y_batch_1)
            # Contour
            y_batch_2 = np.zeros((y_batch_tmp.shape[0], y_batch_tmp.shape[1], y_batch_tmp.shape[2], 2),dtype=K.floatx())
            y_batch_2[:, :, :, 0] = np.add(y_batch_tmp[:, :, :, 0], y_batch_tmp[:, :, :, 1])
            y_batch_2[:, :, :, 0] = y_batch_0[:, :, :, 0] > 0
            y_batch_2[:, :, :, 1] = y_batch_tmp[:, :, :, 2]
            y_batches.append(y_batch_2)

            # for i in range(self.nb_class):
            #    x_batches.append(x_batch_tmp.copy())
        class_weights = None
        if reshape:
            batch_size = y_batch_tmp.shape[0]
            y_batch_tmp = y_batch_tmp.reshape((batch_size, target_size[0] * target_size[1], nb_class))
        if generate_weight:
            class_weights = np.zeros((batch_size, target_size[0] * target_size[1], 3))
            class_weights[:, :, 0] += 0.5
            class_weights[:, :, 1] += 1
            class_weights[:, :, 2] += 1.5

        if reconstruction:
            # print(reconstruction)
            # print("[x_batch_tmp, y_batch_tmp], [y_batch_tmp, x_batch_tmp]")
            if generate_weight:
                yield ([x_batch_tmp, y_batch_tmp, class_weights], [y_batch_tmp, y_batch_tmp])
            else:
                yield ([x_batch_tmp, y_batch_tmp], [y_batch_tmp, y_batch_tmp])
        elif run_one_vs_all_mode:
            if generate_weight:
                yield (x_batch_tmp, y_batches, class_weights)
            else:
                yield (x_batch_tmp, y_batches)
        elif class_to_mask:
            yield ([x_batch_tmp, y_batches], [y_batches, x_batch_tmp])
        else:
            if generate_weight:
                yield (x_batch_tmp, y_batch_tmp, class_weights)
            else:
                yield (x_batch_tmp, y_batch_tmp)


def image_generator(generator, use_cropping=False, input_shape=(256, 256, 3), cropping_size=(128, 128),verbose=False, reconstruction=True, test_mode=False, batch_size=30, number_of_class=2):
    size = cropping_size[0] * cropping_size[1]
    while 1:
        if (verbose == True):
            print("Loading the images...")
        x_batch_tmp, y_batch_tmp = generator.next()
        # np.save("test.np", x_batch_tmp)
        # exit()
        # print(x_batch_tmp.shape)
        if (verbose == True):
            print("Generate the batch images...")

        x_batch_cropped = None
        y_batch_cropped = None
        itm_list_coordination = None
        if (use_cropping):
            b_x_batch_cropped = None
            b_y_batch_cropped = None
            if test_mode:
                coordinations = [(0, 0)]
            else:
                coordinations = ((0, 0),
                                 (int(round(cropping_size[0] / 2.)), 0),
                                 (0, int(round(cropping_size[1] / 2.))),
                                 (int(round(cropping_size[0] / 2.)), int(round(cropping_size[1] / 2.))),
                                 (int(round(cropping_size[0] / 4.)), 0),
                                 (0, int(round(cropping_size[1] / 4.))),
                                 (int(round(cropping_size[0] / 4.)), int(round(cropping_size[1] / 4.))))
            # Generate image patches with different off set
            for coord in coordinations:
                itm_list_coordination = self.CropLayers(input_shape=input_shape, cropping_size=cropping_size,
                                                        off_set=coord)
                # Get the image patches
                for itm in itm_list_coordination:
                    if (b_x_batch_cropped is None):
                        b_x_batch_cropped = x_batch_tmp[:, itm[0]:itm[1], itm[2]:itm[3], :]
                        b_y_batch_cropped = y_batch_tmp[:, itm[0]:itm[1], itm[2]:itm[3], :]
                    else:
                        b_x_batch_cropped = np.append(b_x_batch_cropped,
                                                      x_batch_tmp[:, itm[0]:itm[1], itm[2]:itm[3], :], axis=0)
                        b_y_batch_cropped = np.append(b_y_batch_cropped,
                                                      y_batch_tmp[:, itm[0]:itm[1], itm[2]:itm[3], :], axis=0)

                        # Assign to y_batch_cropped
            x_batch_cropped = b_x_batch_cropped
            y_batch_cropped = b_y_batch_cropped
        else:
            y_batch_cropped = y_batch_tmp
            x_batch_cropped = x_batch_tmp

        # print(x_batch_cropped.shape)
        # Generate batch to deliver
        length_b = x_batch_cropped.shape[0]
        sequence = int(round(length_b / self.args.batch_size))
        if test_mode and use_cropping:
            yield x_batch_cropped, y_batch_cropped, itm_list_coordination
        else:
            for range_x in range(0, sequence):
                begin_index = range_x * self.args.batch_size
                end_index = (range_x + 1) * self.args.batch_size

                if ((range_x + 1) == sequence):
                    end_index = length_b

                x_batch_ = x_batch_cropped[begin_index:end_index, :, :, :]
                y_batch_ = x_batch_cropped[begin_index:end_index, :, :, :]

                difference = end_index - begin_index

                # Fill the gap with random images
                if difference < batch_size:
                    number_itm = difference - batch_size
                    selected_randomly_to_fill_the_gap = np.random.choice(length_b, number_itm)

                    for x_index in selected_randomly_to_fill_the_gap:
                        x_batch_ = np.append(x_batch_, x_batch_cropped[x_index], axis=0)
                        y_batch_ = np.append(y_batch_, x_batch_cropped[x_index], axis=0)

                counter = 0
                zeros_img = []
                ones_img = []
                # Find zeros patchs --> may skew the training effect.
                for xindex in range(0, len(x_batch_) - 1):
                    if (np.count_nonzero(x_batch_[xindex, :, :, 0]) < (size * 0.50)):
                        zeros_img.append(counter)
                    else:
                        ones_img.append(counter)
                    counter = 1 + counter

                # Replace zeros patchs with patch having non-zero pixels in > 50%.
                for x_index_zeros in zeros_img:
                    random_selection_index = np.random.choice(ones_img)
                    x_batch_[x_index_zeros] = x_batch_[random_selection_index]
                    y_batch_[x_index_zeros] = y_batch_[random_selection_index]

                # Generate the labels for these patch images
                y_batch = np.zeros((x_batch_.shape[0], number_of_class), dtype=K.floatx())
                for i, (y) in enumerate(y_batch_):
                    number_of_px = y.shape[0] * y.shape[1]
                    np_flatted = y.flatten()
                    y_indx = np.count_nonzero(np_flatted)
                    batch_yy_class = 0
                    y_per = y_indx / number_of_px
                    if (number_of_class == 3):
                        if (y_per > 0.5):
                            batch_yy_class = 1
                        elif (y_per <= 0.5 and y_per > 0.2):
                            batch_yy_class = 2
                    elif (number_of_class == 2):
                        if (y_per > 0.5):
                            batch_yy_class = 1
                    y_batch[i, batch_yy_class] = 1.

                # x = y_batch_cropped - np.mean(y_batch_cropped)
                # Center by mean
                # Calculate Prinicpal components...
                # x = x_batch_cropped
                # flat_x = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
                # sigma = np.dot(flat_x.T, flat_x) / flat_x.shape[0]
                # u, s, _ = linalg.svd(sigma)
                # principal_components = np.dot(np.dot(u, np.diag(1. / np.sqrt(s + 1e-6))), u.T)
                # if principal_components is not None:
                #    flatx = np.reshape(x, (-1, np.prod(x.shape[-3:])))
                # whitex = np.dot(flatx, principal_components)
                # x_batch_cropped = np.reshape(whitex, x_batch_cropped.shape)
                # print(x_batch)
                # print(y_batch)
                # print(x_batch.shape)
                if (verbose == True):
                    print(range_x)
                if reconstruction:
                    yield ([x_batch_[:, :, :, 0:2], y_batch], [y_batch, x_batch_[:, :, :, 0:2]])  # y_batch_[:,:,:,0]])
                else:
                    yield ([x_batch_, y_batch])


#End General function
class OpenSlideOnlivePatch:
    def __init__(self, image_folder, mask_folder=None, select_criteria=None, clipped=False, zooming_level=40, target_zooming=40, annotation_zooming_level=10, split_text_annotation=""):
        '''
        :param image_folder: Define the directory for extracted images. This directory should exist before running any function in this class! -Mandatory parameter-
        :param mask_folder: Define the directory for generated mask. This directory should exist before running any function in this class!
        :param select_criteria: Set specific criteria like keywords e.g. T3, T4 depending on the annotation codes
        :param clipped: Consider only the histology structure inside the mask. All values outside the mask will set to 0
        :param zooming_level: Define the zooming level of the histology image
        :param target_zooming: Set the target zooming level. It should be equal or lesser than the zooming level.
        :param annotation_zooming_level: Set the zooming level, at which the annotation was made. Default: 10
        :param split_text_annotation: Define the text or char to split the context of the label. Default: ""
        '''
        self.zooming_level = zooming_level
        self.split_text_annotion = split_text_annotation
        self.select_criteria = select_criteria
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.annotation_zooming_level = annotation_zooming_level
        self.clipped = clipped
        self.target_zooming = target_zooming
        self.factor_for_mask = self.target_zooming / self.annotation_zooming_level
        self.factor = self.target_zooming / self.zooming_level


    def GetTheMaxSizeOfRectangleOfEachPolygons(self, polygons):
        RectMaxOfEachPolygons = []
        for poly in polygons:
            X_min = np.min(poly[:, :, 0])
            X_max = np.max(poly[:, :, 0])
            Y_min = np.min(poly[:, :, 1])
            Y_max = np.max(poly[:, :, 1])
            width_ = X_max - X_min
            Height_ = Y_max - Y_min
            # Save in format compatible to OpenSlide
            RectMaxOfEachPolygons.append([X_min, Y_min, width_, Height_])
        return RectMaxOfEachPolygons

    def GetImagePatchOnlive(self, image, regionofinterest, level=0):
        '''
        :param image: Openslide object (You create this object when you call LoadImage.
        :param regionofinterest: Define the rectangle of the region of interest [X,Y, width, height]
        :param level: Determine the image level you want to consider. O = the lowest level.
        :return: A PIL Image
        '''
        regionofinterest_T = regionofinterest * self.factor_for_mask
        regionofinterest_T = np.around(regionofinterest_T, decimals=0)
        org_img = image.read_region(location=(regionofinterest_T[0], regionofinterest_T[1]), level=0,
                                    size=(regionofinterest_T[2], regionofinterest_T[3]))
        crop_img = np.asarray(org_img)
        return crop_img

    def LoadImage(self, filename):
        '''
        :param filename: Give a filename of supported images.
        :return:
        '''
        print("Proce: Loading the file ",filename)
        try:
            self.image = OpenSlide(filename)
            # temp = str(image.properties)[14:-1].replace("u'", "\"")
            # temp = temp.replace("'", r'"')
            # temp = json.loads(temp)
        except OSError:
            print(OSError.message)
        return self.image

    def LoadMask(self, filename, return_identity_list=False):
        LoadAll = False
        if self.select_criteria is None:
            LoadAll = True

        polygons = self.ExtractPolygons(ET.parse(filename).getroot(), return_identity_list=return_identity_list, LoadAll=LoadAll, select_criteria=self.select_criteria)
        return polygons

    def _GettissueArea(self,level=3):
        heighest_level = len(self.image.level_dimensions) -1
        level = heighest_level
        print(level)
        print(self.image.level_dimensions[level])
        image = self.image.read_region((0, 0), level, self.image.level_dimensions[level])
        image = np.asarray(image)
        image = image[:, :, 0:3]
        plt.imshow(image)
        HE = color.rgb2hed(image)
        tissues = HE[:, :, 0]
        plt.imshow(HE)
        plt.show()
        from skimage.filters import threshold_minimum
        thresh_min = threshold_minimum(tissues)
        binary_min = tissues <= thresh_min
        plt.imshow(binary_min)
        plt.show()
        return binary_min

    def old_GettissueArea(self, level=3):
        heighest_level = len(self.image.level_dimensions) - 1
        level = heighest_level
        image = self.image.read_region((0, 0), level, self.image.level_dimensions[level])
        image = np.asarray(image)
        image = image[:, :, 0:3]
        image = skimage.color.rgb2grey(image)
        from skimage.filters import threshold_otsu
        image = filters.gaussian(image, 2)
        thresh_min = threshold_otsu(image)
        binary_min = image <= thresh_min
        tissues = binary_min
        tissues = morphology.closing(tissues, square(2))
        tissues = morphology.opening(tissues, square(2))
        #cv2.imwrite("./tissue.png", tissues * 255.)
        return tissues


    def MaxRegion(self, regions):
        rx = None
        for region in regions:
            if rx is None:
                rx = region
            elif rx.area < region.area:
                rx = region
        return rx

    def LoadTissueMask(self):
        '''
        Determine the tissue surface.
        :return: mask, rectangle (Y,X, Height, Width) = (row, columne, rows depth, columne depth)
        '''
        print("Proc: Determine tissue area...")
        level_3 = self._GettissueArea(3)
        plt.imshow(level_3)
        plt.show()
        level = level_3
        cleared = clear_border(level)
        label_image = label(cleared)
        regions = regionprops(label_image)
        heighest_level = len(self.image.level_dimensions) - 1
        level_factor = np.divide(self.image.level_dimensions[0], self.image.level_dimensions[heighest_level])

        region = self.MaxRegion(regions)
        minr, minc, maxr, maxc = region.bbox
        minr_new = minr * level_factor
        minc_new = minc * level_factor

        maxr_new = maxr * level_factor
        maxc_new = maxc * level_factor
        rect = (np.around((minr_new[1], minc_new[0], maxr_new[1] - minr_new[1], maxc_new[0] - minc_new[0]),
                         decimals=0)).astype(np.int)
        print(rect)

        shape_To_convert = (rect[2], rect[3])
        print(shape_To_convert)

        level_0 = region.image #skimage.transform.resize(region.image, shape_To_convert)
        #cv2.imwrite("./test.png", level_0*255.)
        print("Done: Determine tissue area...")
        return level_0,  rect, level_factor[0]

    def GetLesionIdentity(self, xml_root, AttributeParameter="Description", AttributeValue="LCM"):
        '''
        :param xml_root:
        :param AttributeParameter: define the attribute name to be looked in (XML Element Attributes in SVS annotation file)
        :param AttributeValue: Select only items ordered under this value.
        :return: a list of lesion identity
        '''
        annots = xml_root
        id_list = []
        finding_name = ""
        Go_next_step = False
        for annot in annots:
            for child in annot:
                if child.tag == 'Attributes':
                    c_d = child
                    for at in c_d:
                        if (at.tag == 'Attribute'):
                            if (AttributeParameter in at.attrib):
                                m_value = at.attrib['Value']
                                if m_value.lower() == AttributeValue.lower():
                                    finding_name = AttributeValue
                                    break
                                else:
                                    finding_name = ""

                if child.tag == 'Regions':
                    regions = child
                    for region in regions:
                        if region.tag == 'Region':
                            text_description = region.attrib['Text']
                            # Add the attribute from the major location
                            if (text_description == ""):
                                text_description = finding_name
                            id_list.append(text_description)
        return id_list

    def ExtractPolygons(self, xml_root, LoadAll=True, LoadOnlyPoints=True,AttributeParameter="Description", IgnoreLesionsWithNoDescription=True, AttributeValue="LCM", select_criteria=None, return_identity_list=False):
        '''
        :param xml_root:
        :param LoadAll:
        :param factor_zooming:
        :param LoadOnlyPoints:
        :param AttributeParameter:
        :param AttributeValue:
        :param select_criteria: <- lesion id
        :param return_identity_list: set true if you want to have the identity of the lesion as well.
        :return: list of points for each area.
        '''
        annots = xml_root
        polys = []
        identity_list = []
        finding_name = ""
        Go_next_step = False
        for annot in annots:
            for child in annot:
                if child.tag == 'Attributes':
                    c_d = child
                    for at in c_d:
                        if (at.tag == 'Attribute'):
                            if (AttributeParameter in at.attrib):
                                m_value = at.attrib['Value']
                                if m_value.lower() == AttributeValue.lower():
                                    finding_name = AttributeValue
                                    break
                                else:
                                    finding_name = ""

                if child.tag == 'Regions':
                    regions = child
                    for region in regions:
                        if region.tag == 'Region':
                            text_description = region.attrib['Text']
                            identity_list.append(text_description)
                            # Add the attribute from the major location
                            if IgnoreLesionsWithNoDescription and text_description == "":
                                continue

                            if (text_description == ""):
                                text_description = finding_name
                            list_of_words_to_look = [text_description]
                            if (self.split_text_annotation != ""):
                                parameters = text_description.split(self.split_text_annotation)
                                list_of_words_to_look = parameters
                                # Debug: print(list_of_words_to_look)

                            for word_to_look in list_of_words_to_look:
                                if (word_to_look.lower() in select_criteria and LoadAll == False):
                                    # Debug: print("Looking for: " + word_to_look.lower())
                                    for child in region:
                                        if child.tag == 'Vertices':
                                            vertices = child
                                            pts = []
                                            for vertex in vertices:
                                                pts.append([int(vertex.attrib['X']), int(vertex.attrib['Y'])])
                                            pts = np.array([pts], np.int32)

                                            if (LoadOnlyPoints):
                                                polys.append(pts)
                                            else:
                                                print("Not emplemented...")

                            if (LoadAll):
                                for child in region:
                                    if child.tag == 'Vertices':
                                        vertices = child
                                        pts = []
                                        for vertex in vertices:
                                            pt_X = int(round(float(vertex.attrib['X'])))
                                            pt_Y = int(round(float(vertex.attrib['Y'])))
                                            pts.append([pt_X, pt_Y])
                                        pts = np.array([pts], np.int32)

                                        if (LoadOnlyPoints):
                                            polys.append(pts)
                                        else:
                                            print("Not emplemented...")
                            else:
                                continue
        if return_identity_list:
            return polys, identity_list
        else:
            return polys


    def RandomRegionDefinition(self, mask, offset_coordination, max_patch_number, patch_size, factor=1):
        '''
        :param mask: mask image
        :param max_patch_number: Define the batch size
        :param patch_size: Define the patch dimension
        :return: A list of X,Y coordinations randomly defined.
        '''
        print("Proc: Random region definition")
        print(offset_coordination)
        dimension = mask.shape
        reg_lst = []
        counter = 0
        #plt.imshow(mask)
        #plt.show()
        total_size = patch_size[0] * patch_size[1]
        tolerance = 500
        count = 0
        patch_x_length =  int(round(patch_size[0] / factor))
        patch_y_length = int(round(patch_size[1] / factor))
        while counter < max_patch_number:
            count = count + 1
            x = random.randint(0,  dimension[1] - patch_x_length)
            y = random.randint(0, dimension[0] - patch_y_length)

            mask_selected = mask[y:y + patch_y_length, x:x + patch_x_length]
            number_positive = np.count_nonzero(mask_selected)

            percentage_positive = number_positive / total_size
            if percentage_positive > 0.80:
                x = (x * factor) + offset_coordination[1]
                y = (y * factor) + offset_coordination[0]

                img = self.image.read_region((x,y),0,patch_size)
                image = np.asarray(img)
                image = image[:, :, 0:3]
                HE = color.rgb2hed(image)
                tissues = HE[:, :, 0]
                from skimage.filters import threshold_minimum
                thresh_min = threshold_minimum(tissues)
                binary_min = tissues <= thresh_min
                number_positive = np.count_nonzero(binary_min)
                percentage_positive = number_positive / total_size
                if percentage_positive > 0.80:
                    reg_lst.append([x, y])
                    counter = counter + 1
            if count >= tolerance:
                print("Stopped after 500 tries...")
                break

        print("Done: Random region definition")
        return reg_lst

    def GetPatches(self, image, mask, batch_size, patch_size, coordination):
        '''
        :param image: the original image
        :param mask:  the mask (2-Dimension boolean)
        :param batch_size: Define the batch size for this image 100
        :param patch_size: the dimension of the patch (512, 512)
        :return: patched images in numpy array (batch_size, height, width, channels.
        '''
        regions = self.RandomRegionDefinition(mask, coordination, batch_size, patch_size)
        patch_imgs = np.zeros((batch_size, patch_size[0], patch_size[1], image.shape[2]), dtype=np.uint8)

        for index, region in enumerate(regions):
            x, y = region
            patch_imgs[index] = image[y:y + patch_size[1], x:x + patch_size[0]].copy()
        return patch_imgs

    def GetPatchesAsFiles(self,mask, patch_per_image, patch_size, filename, type_data="train", n_class="", coordination=(0,0,0,0), factor=0):
        regions = self.RandomRegionDefinition(mask, coordination, patch_per_image, patch_size, factor)
        print("Proc: Generate patch images....")
        file_ex = os.path.basename(filename)
        file_to_use = os.path.splitext(file_ex)[0]
        n_class = int(round(n_class,0))
        x_file_path = self.image_folder + "/%s/%s/" % (type_data,n_class)
        counter = 0
        str_input_path = "/%s/%s/" % (type_data, n_class)
        make_folders(self.image_folder, str_input_path)
        for index, region in enumerate(regions):
            x, y = region
            tmp_x_file_path = x_file_path + "%s_%s.png" %(file_to_use, counter)
            counter += 1
            img = self.image.read_region((x,y),0,patch_size)
            img.save(tmp_x_file_path)
        print("Done: Patch images")


    # Begin: Major functions (Also an example use for other functions listed above)
    def CreatePatchesToStoreAsFiles(self, filename, annotation_file):
        '''
        :param filename: the file path for the histology images supported by OpenSlide
        :param annotation_file: the annotation file path: It should be a XML file stored in a ImageScope format
        :param Debug: Not active
        :return: A log_report in numpy format.
        '''
        img_dir_to_store = self.image_folder
        mask_dir_to_store = self.mask_folder

        OpenSlide_obj = self.LoadImage(filename)
        file_ex = os.path.basename(filename)
        file_to_use = os.path.splitext(file_ex)[0]
        polygons, identities = self.LoadMask(annotation_file, return_identity_list=True)
        rectangles = self.AnalyseTheMaxSizeOf(polygons)
        log_list_of_files = []
        counter = 0.
        for rect, polygon, id_itm in zip(rectangles, polygons, identities):
            # Define the filename
            file_name_to_store = file_to_use + "_" + id_itm + ".png"

            # Create mask
            # Numpy dim format: Height, width --> Check this, it works weird sometimes.

            # Scale the polygon and the mask image
            rect_new = rect * self.factor_for_mask
            rect_new = np.around(rect_new, decimals=0)
            maskimage = np.zeros((rect_new[3], rect_new[2]), dtype=np.uint8)

            polygon_new = polygon * self.factor_for_mask
            polygon_new = np.around(polygon_new, decimals=0)
            cv2.fillPoly(maskimage, polygon_new, color=1)
            fln_mask = None

            if mask_dir_to_store is not None:
                fln_mask = os.path.join(img_dir_to_store, file_name_to_store)
                cv2.imwrite(fln_mask, maskimage)
            else:
                if (counter == 0):
                    print("Warning: Mask directory is not defined. Mask files will be not saved...")
                    counter += 1

            # Get image path and store
            img = self.GetImagePatchOnlive(OpenSlide_obj, rect, level=0)

            pix = np.array(img)
            if self.clipped:
                for i in range(0, 3, 1):
                    pix[:, :, i] = pix[:, :, i] * maskimage
            # Consider only the first three channels
            pix = pix[:, :, 0:3]
            img = Image.fromarray(pix)
            img_filename_patch = os.path.join(img_dir_to_store, file_name_to_store)
            img.save(img_filename_patch)
            # Include Case Id, lesion Id, filename, path for the patch, path of the mask file
            log_list_of_files.append([file_to_use, id_itm, file_name_to_store, img_filename_patch, fln_mask])
        return log_list_of_files

    def GeneratePatchesAsNumpy(self, filename, annotation_file):
        '''
        :param filename: A complete file path for the histology images supported by OpenSlide
        :param annotation_file: A complete  path for the annotation file: It should be a XML file stored in a ImageScope format
        :param img_dir_to_store: Define the directory to store the image
        :param mask_dir_to_store:  Define the mask directory to store the mask
        :param Debug: Not active
        :return: A log_report in numpy format.
        '''
        OpenSlide_obj = self.LoadImage(filename)
        file_ex = os.path.basename(filename)
        file_to_use = os.path.splitext(file_ex)[0]
        polygons, identities = self.LoadMask(annotation_file, return_identity_list=True)
        rectangles = self.AnalyseTheMaxSizeOf(polygons)
        data_storage = []
        for rect, polygon, id_itm in zip(rectangles, polygons, identities):
            # Create mask
            # Numpy dim format: Height, width <- Check this using plt.imshow()
            # Scale the polygon and the mask image
            rect_new = rect * self.factor_for_mask
            rect_new = np.around(rect_new, decimals=0)
            maskimage = np.zeros((rect_new[3], rect_new[2]), dtype=np.uint8)

            polygon_new = polygon * self.factor_for_mask
            polygon_new = np.around(polygon_new, decimals=0)
            cv2.fillPoly(maskimage, polygon_new, color=1)

            # Get image path and store
            img = self.GetImagePatchOnlive(OpenSlide_obj, rect, level=0)

            pix = np.array(img)
            if self.clipped:
                for i in range(0, 3, 1):
                    pix[:, :, i] = pix[:, :, i] * maskimage
            # Consider only the first three channels
            pix = pix[:, :, 0:3]

            data_storage.append([file_to_use, id_itm, pix, maskimage])
        return data_storage

    def GeneratePatchDirect(self, filename, patch_size=(512,512), batch_size=256):
        self.LoadImage(filename)
        level_0, region_def, factor = self.LoadTissueMask()
        image = self.image.read_region((region_def[1], region_def[0]), level=0, size=(region_def[3], region_def[2]))
        patch_images = self.GetPatches(image, level_0, batch_size, patch_size, region_def)
        return patch_images

    def GeneratePatchDirectAsImagePatch(self, filename, patch_size=(512,512), batch_size=128, type_data="", n_class=""):
        self.LoadImage(filename)
        level_0, region_def, factor = self.LoadTissueMask()
        self.GetPatchesAsFiles(level_0, batch_size, patch_size, filename, type_data, n_class, region_def, factor)
    # End Major functions

class Filedirectoryamagement():
    def __init__(self, mask_directories, image_directories, image_extension=[".svs", ".tiff"], mask_extension=[".xml"]):
        '''
        :param mask_directories:
        :param image_directories:
        :param image_extension:
        :param mask_extension:
        :return:
        '''
        self.mask_directories = mask_directories
        self.image_directories = image_directories
        self.img_extension = image_extension
        self.mask_extension = mask_extension
        self.files = None

    def FindMatchedImagesAndAnnotation(self, list_of_image_files, list_of_xml_files):
        image_matched_xml = {}
        for file_xml in list_of_xml_files:
            # print(list_of_xml_files[file_xml])
            str_file = os.path.basename(file_xml)
            str_file = os.path.splitext(str_file)[0]
            for str_img_fl in list_of_image_files:
                str_img_file = os.path.basename(str_img_fl)
                str_img_file = os.path.splitext(str_img_file)[0]
                if str_file == str_img_file:
                    image_matched_xml[list_of_xml_files[file_xml]] = list_of_image_files[str_img_fl]
                    continue
        return image_matched_xml

    def GenerateFileListFromDirectory(self, directories, extension):
        list_of_files = {}
        #print(directories)
        #for dir_c in directories:
        #print(dir_c)
        if (os.path.exists(directories)):
            for (dirpath, dirnames, filenames) in os.walk(directories):
                for filename in filenames:
                    # print(extension)
                    if filename.endswith(tuple(extension)):
                        #print(filename)
                        list_of_files[filename] = os.sep.join([dirpath, filename])
        return list_of_files

    def LoadFiles(self, Mask=True):
        img_files = self.GenerateFileListFromDirectory(self.image_directories, self.img_extension)
        if Mask:
            mask_files = self.GenerateFileListFromDirectory(self.mask_directories,self.mask_extension)
            self.files = self.FindMatchedImagesAndAnnotation(img_files, mask_files)
        else:
            self.files = img_files
        files_keys =list(self.files.keys())
        random.shuffle(files_keys)

        file_shuffle = {}
        for key in files_keys:
            file_shuffle[key] = self.files[key]
        self.files = file_shuffle
        #print(self.files)

    def GenerateKFoldValidation(self, k=5):
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=k)
        k_fold_validation_files = []
        for train, test in kf.split(self.files.keys()):
            k_fold_validation_files.append([train, test])
        return k_fold_validation_files

    def GetFilename(self, sample_id):
        for fln in self.files[0]:
            str_file = os.path.basename(fln)
            str_file = os.path.splitext(str_file)[0]
            if sample_id == str_file[0:15]:
                return fln
        return None

    def generate_patch_images_train_valid_test(self, patch_per_image=100, image_patch_size=(512,512), subset_ratio=[0.7,0.6,0.4], directory="", class_filename="", type_class_col="sample_type"):
        print("Generate file list...")
        self.LoadFiles(Mask=False)

        import pandas as pd
        print("Loading the classification file...")
        data = pd.read_csv(class_filename, index_col=0)

        files_indexed = {}
        for file in self.files.keys():
            str_file = os.path.splitext(file)[0]
            sample_id = str_file[0:15]
            sample_id = sample_id.replace("-", ".")
            files_indexed[sample_id] = self.files[file]


        #Remove rows not having the image file
        data = data.loc[files_indexed.keys(),:]

        #Get the classes
        list_of_unique_value = pd.Series(data.CNV_Status, name=type_class_col).unique()


        files_groups = {}
        for x in list_of_unique_value:
            files_groups[x] = data.loc[data[type_class_col] == x, type_class_col]

        train_files = []
        test_files = []
        valid_files = []
        for group in files_groups.keys():
            sample_list = files_groups[group]
            train_length = int(round(len(files_groups[group]) * subset_ratio[0]))
            remaining = len(files_groups[group]) - train_length

            test_length = int(round(remaining * subset_ratio[1]))
            valid_length = remaining - test_length

            train_files.extend(random.sample(list(sample_list.index), train_length))
            remaining = [e for e in list(sample_list.index) if e not in train_files]
            test_files.extend(random.sample(list(remaining), test_length))
            valid_files.extend([e for e in remaining if e not in test_files])

        print("number of samples for train set" , len(train_files))
        print("number of samples for test set", len(test_files))
        print("number of samples for validation set", len(valid_files))

        files_subsection = {'train':train_files, 'test': test_files, 'valid': valid_files}
        bdx = OpenSlideOnlivePatch(image_folder=directory)

        #Determine the subclass to generate the directories.
        print("subclasses are ", list_of_unique_value)


        for type_data in files_subsection.keys():
            print("Starting with: ", type_data)
            filelist = files_subsection[type_data]
            print("Selected samples:")
            print(filelist)
            for sample_id in filelist:
                if (sample_id in files_indexed):
                    filename = files_indexed[sample_id]
                    class_type = data.loc[sample_id, type_class_col]
                    bdx.GeneratePatchDirectAsImagePatch(filename, image_patch_size, patch_per_image, type_data, class_type)
                else:
                    print("No file was found for ", sample_id)

    def image_generator_flow(self, generator, reconstruction=True, verbose=False, classify=False, spare_category=False,
                             generate_weight=False, reshape=False, run_one_vs_all_mode=False, weights=None):
        while 1:
            if (verbose == True):
                print("Loading the images...")
            x_batch_tmp, y_batch_tmp = generator.next()

            if (verbose == True):
                print("Generate the batch images...")

            if classify:
                y_batch = np.zeros((y_batch_tmp.shape[0], self.nb_class), dtype=K.floatx())
                for i, (y) in enumerate(y_batch_tmp):
                    number_of_px = y.shape[0] * y.shape[1]
                    np_flatted = y.flatten()
                    y_indx = np.count_nonzero(np_flatted)
                    batch_yy_class = 0
                    y_per = y_indx / number_of_px

                    if (self.nb_class == 2):
                        if (y_indx > 0):
                            batch_yy_class = 1
                    # print(batch_yy_class)
                    y_batch[i, batch_yy_class] = 1.
            y_batches = []
            if spare_category:
                pass

            if run_one_vs_all_mode:
                # Background
                y_batch_0 = np.zeros((y_batch_tmp.shape[0], y_batch_tmp.shape[1], y_batch_tmp.shape[2], 2),
                                     dtype=K.floatx())
                y_batch_0[:, :, :, 0] = np.add(y_batch_tmp[:, :, :, 1], y_batch_tmp[:, :, :, 2])
                y_batch_0[:, :, :, 0] = y_batch_0[:, :, :, 0] > 0
                y_batch_0[:, :, :, 1] = y_batch_tmp[:, :, :, 0]
                y_batches.append(y_batch_0)
                # Nucleous
                y_batch_1 = np.zeros((y_batch_tmp.shape[0], y_batch_tmp.shape[1], y_batch_tmp.shape[2], 2),
                                     dtype=K.floatx())
                y_batch_1[:, :, :, 0] = np.add(y_batch_tmp[:, :, :, 0], y_batch_tmp[:, :, :, 2])
                y_batch_1[:, :, :, 0] = y_batch_0[:, :, :, 0] > 0
                y_batch_1[:, :, :, 1] = y_batch_tmp[:, :, :, 1]
                y_batches.append(y_batch_1)
                # Contour
                y_batch_2 = np.zeros((y_batch_tmp.shape[0], y_batch_tmp.shape[1], y_batch_tmp.shape[2], 2),
                                     dtype=K.floatx())
                y_batch_2[:, :, :, 0] = np.add(y_batch_tmp[:, :, :, 0], y_batch_tmp[:, :, :, 1])
                y_batch_2[:, :, :, 0] = y_batch_0[:, :, :, 0] > 0
                y_batch_2[:, :, :, 1] = y_batch_tmp[:, :, :, 2]
                y_batches.append(y_batch_2)

                # for i in range(self.nb_class):
                #    x_batches.append(x_batch_tmp.copy())
            class_weights = None
            if reshape:
                batch_size = y_batch_tmp.shape[0]
                y_batch_tmp = y_batch_tmp.reshape(
                    (batch_size, self.target_size[0] * self.target_size[1], self.nb_class))
            if generate_weight:
                class_weights = np.zeros((self.args.batch_size, self.target_size[0] * self.target_size[1], 3))
                class_weights[:, :, 0] += 0.5
                class_weights[:, :, 1] += 1
                class_weights[:, :, 2] += 1.5

            if reconstruction:
                # print(reconstruction)
                # print("[x_batch_tmp, y_batch_tmp], [y_batch_tmp, x_batch_tmp]")
                if generate_weight:
                    yield ([x_batch_tmp, y_batch_tmp, class_weights], [y_batch_tmp, y_batch_tmp])
                else:
                    yield ([x_batch_tmp, y_batch_tmp], [y_batch_tmp, y_batch_tmp])
            elif run_one_vs_all_mode:
                if generate_weight:
                    yield (x_batch_tmp, y_batches, class_weights)
                else:
                    yield (x_batch_tmp, y_batches)
            else:
                if generate_weight:
                    yield (x_batch_tmp, y_batch_tmp, class_weights)
                else:
                    yield (x_batch_tmp, y_batch_tmp)





