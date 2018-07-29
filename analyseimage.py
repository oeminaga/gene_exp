import keras
import PIL
import cv2
import numpy as np
import utils
import keras.backend as K
from matplotlib import pyplot as plt
import math

def PredictImage(model, filename, case_id, filename_to_save_heatmap, zooming_level, normalization_color="OD",
                 number_of_channel=3, target_size=(64,64), nb_class=2, batch_size=16):
    _PIL_INTERPOLATION_METHODS = {
        'nearest': PIL.Image.NEAREST,
        'bilinear': PIL.Image.BILINEAR,
        'bicubic': PIL.Image.BICUBIC,
    }

    # These methods were only introduced in version 3.4.0 (2016).
    if hasattr(PIL.Image, 'HAMMING'):
        _PIL_INTERPOLATION_METHODS['hamming'] = PIL.Image.HAMMING

    if hasattr(PIL.Image, 'BOX'):
        _PIL_INTERPOLATION_METHODS['box'] = PIL.Image.BOX
    # This method is new in version 1.1.3 (2013).

    if hasattr(PIL.Image, 'LANCZOS'):
        _PIL_INTERPOLATION_METHODS['lanczos'] = PIL.Image.LANCZOS

    print("Loading the image...")
    print(filename)
    print(filename_to_save_heatmap)
    img_org = cv2.imread(filename)

    img_org = utils.EnhanceColor(img_org)
    # img_org = skimage.filters.gaussian(img_org, 1, multichannel=True)
    # plt.imshow(img_org)
    # plt.show()
    img_org = np.asarray(img_org)
    # plt.imshow(img_org)
    # plt.show()
    if normalization_color == "HE":
        print("Using HE channels...")
        img_org = utils.Convert_to_HRD(img_org)

    elif normalization_color == "HSV":
        print("Proc: Appyling HSV")
        img_org = utils.Convert_to_HSV(img_org)

    elif normalization_color == "OD":
        print("Using OD channels...")
        img_org = utils.Convert_to_OD(img_org)

    else:
        print("Using RGB channels...")
        img_org = utils.LN_normalize(img_org)
        # simg_org =img_org * 1./255.

    print("Run the heatmap generation...")

    regions_data = {"regions": (0, 0),
                    "regions_o_50": (0, int(round(target_size[1] * 0.5))),
                    "regions_l_50": (int(round(target_size[0] * 0.5)), 0),
                    "regions_w_50": (int(round(target_size[0] * 0.5)), int(round(target_size[1] * 0.5))),
                    # "regions_l_25": (0, int(round(self.target_size[0]*0.25))),
                    # "regions_w_25": (int(round(self.target_size[0]*0.25)), 0),
                    # "regions_o_25": (int(round(self.target_size[0]*0.25)), int(round(self.target_size[0]*0.25))),
                    # "regions_l_75": (int(round(self.target_size[0]*0.75)), int(round(0))),
                    # "regions_o_75": (int(round(self.target_size[0]*0.75)), int(round(self.target_size[0]*0.75))),
                    # "regions_w_75": (int(round(0)), int(round(self.target_size[0]*0.75))),
                    }

    # Generate the platform...
    levels_regions = {}
    levels_batches = {}
    heatmap_tmp_list = {}
    counter_nb_regions = 0
    for region_name in regions_data.keys():
        levels_regions[region_name] = utils.CropLayers(input_shape=img_org.shape,
                                                      cropping_size=(target_size[0], target_size[1]),
                                                      off_set=regions_data[region_name])
        levels_batches[region_name] = np.zeros(
            (len(levels_regions[region_name]),) + (target_size[0], target_size[1], number_of_channel),
            dtype=K.floatx())
        heatmap_tmp_list[region_name] = np.zeros((img_org.shape[0], img_org.shape[1], nb_class), dtype=np.uint8)
        counter_nb_regions = counter_nb_regions + len(levels_regions[region_name])

    print("Number of regions:" + str(counter_nb_regions))
    print("Predicting the location of tumor lesions...")
    print("Eight-level determination...")

    # Generate the template..
    heatmap_tmp = np.zeros((img_org.shape[0], img_org.shape[1], nb_class), dtype='float')
    heatmap_normalizer = np.zeros(
        (img_org.shape[0], img_org.shape[1]), dtype='float')

    # generates patches from the images from different locations.
    for level in regions_data.keys():
        # 1. Split in regions
        print("Doing: " + level)
        # counter = 0
        for counter, region in enumerate(levels_regions[level]):
            levels_batches[level][counter] = img_org[region[0]:region[1], region[2]: region[3], :].copy()
            # counter = counter +1
        print(levels_batches[level].shape)
        # 2. Predict
        print("Proc: Run the prediction")
        heatmap_predicted_y_x = model.predict(levels_batches[level], batch_size=int(round(batch_size * 0.33)),
                                              verbose=1)
        heatmap_predicted_y_x = heatmap_predicted_y_x.reshape(levels_batches[level].shape)
        print("Proc: Reconstruct the heatmap as png image file...")
        # 3. Reconstruct from the used images

        for x_img, region in zip(heatmap_predicted_y_x, levels_regions[level]):
            predicted_image_tmp = np.ones((target_size[0], target_size[1], nb_class), dtype='float')
            predicted_image_tmp = (predicted_image_tmp * x_img)
            predicted_img_tmp = np.around(predicted_image_tmp, decimals=0)
            heatmap_tmp[region[0]:region[1], region[2]: region[3]] += predicted_img_tmp  # * 255
            heatmap_normalizer[region[0]:region[1], region[2]: region[3]] += 1

    # 4. add the values into the general heatmap
    # heatmap_tmp= (heatmap_tmp_list[level] + heatmap_tmp)
    heatmap_tmp[:, :, 0] = heatmap_tmp[:, :, 0] / heatmap_normalizer
    heatmap_tmp[:, :, 1] = heatmap_tmp[:, :, 1] / heatmap_normalizer
    heatmap_tmp[:, :, 2] = heatmap_tmp[:, :, 2] / heatmap_normalizer

    np.save("./heatmap_tmp.npy", heatmap_tmp)
    # 5. Clean the edges
    '''
    heatmap_tmp[heatmap_tmp<0.] = 0.
    heatmap_tmp[heatmap_tmp>1.] = 1.

    heatmap_tmp[heatmap_tmp[:,:,0]>0.5,1] = 0
    heatmap_tmp[heatmap_tmp[:,:,0]>0.5,2] = 0
    heatmap_tmp[heatmap_tmp[:,:,0]>0.5,0] = 1

    heatmap_tmp[heatmap_tmp[:,:,2]>0.5,1] = 0
    heatmap_tmp[heatmap_tmp[:,:,2]>0.5,2] = 1
    heatmap_tmp[heatmap_tmp[:,:,2]>0.5,0] = 0

    heatmap_tmp[heatmap_tmp[:,:,1]>0.5,1] = 1
    heatmap_tmp[heatmap_tmp[:,:,1]>0.5,2] = 0
    heatmap_tmp[heatmap_tmp[:,:,1]>0.5,0] = 0
    '''

    print(np.min(heatmap_tmp))
    print(np.max(heatmap_tmp))
    '''
    kernel =  np.ones((8,8),np.uint8)
    heatmap_tmp[:,:,1] = cv2.morphologyEx(heatmap_tmp[:,:,1], cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((4, 4), np.uint8)
    heatmap_tmp[:,:,1] = cv2.morphologyEx(heatmap_tmp[:,:,1], cv2.MORPH_OPEN, kernel)
    '''
    plt.imshow(heatmap_tmp)
    plt.show()
    heatmap_tmp_color = heatmap_tmp.copy()
    heatmap_tmp_color = heatmap_tmp_color[:, :, 0] < 0.3  # * img_org[:,:,1] * 255.
    np.save("./heatmap_tmp_color.npy", heatmap_tmp_color)

    # 6. Clean the images
    from skimage.measure import label, regionprops
    from skimage.color import label2rgb
    import matplotlib.patches as mpatches
    from scipy import ndimage as ndi
    from skimage import morphology

    img_org_tmp = img_org.copy() * 255
    img_rgb = cv2.cvtColor(img_org_tmp.astype(np.uint8), cv2.COLOR_HSV2RGB)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    np.save("./heatmap_tmp_gray.npy", gray)

    gray = gray * heatmap_tmp_color
    edges = cv2.Canny(gray, 140, 220)
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel, iterations=2)
    # edges = cv2.dilate(edges,kernel,iterations = 2)
    # minima = extrema.h_minima(heatmap_tmp_color, 40)
    # heatmap_tmp[:, :, 2] = (heatmap_tmp[:, :, 2] + minima) / 2
    # heatmap_tmp[:, :, 2] = heatmap_tmp[:, :, 2] >0.
    heatmap_tmp[:, :, 2] = (heatmap_tmp[:, :, 2] + edges) / 2

    plt.imshow(edges)
    plt.show()

    plt.imshow(heatmap_tmp)
    plt.show()

    img_x = (img_org + heatmap_tmp) / 2.

    img = img_x[:, :, 1]
    img = np.uint8(np.around(img * 255, decimals=0))

    # img = np.float32(img_x)
    corners = cv2.cornerHarris(img, 3, 3, 0.0001)
    heatmap_tmp[:, :, 2] = heatmap_tmp[:, :, 2] + corners
    plt.imshow(corners)
    plt.show()
    plt.imshow(heatmap_tmp)
    plt.show()
    np.save("./heatmap_tmp_corners.npy", corners)
    # 7. Detect the spots
    mask_0_1 = heatmap_tmp[:, :, 1] > 0
    plt.imshow(mask_0_1)
    plt.show()
    import scipy
    label_image = label(mask_0_1)
    image_label_overlay = label2rgb(label_image, image=img)
    region_number = len(regionprops(label_image))
    number_positive_pixel = np.count_nonzero(mask_0_1)
    average_pixel_size = number_positive_pixel / region_number

    # Get Info about the spots
    labeled_mask = np.zeros((heatmap_tmp.shape[0], heatmap_tmp.shape[0]), dtype=np.int)
    counter = 0
    get_average_perimeter = 0
    min_region = 999999999999
    max_region = 0
    list_area = []
    list_perimeter = []
    list_area = []
    list_perimeter = []
    for region in regionprops(label_image):
        if region.area >= 10:
            get_average_perimeter += region.perimeter
        if min_region > region.area:
            min_region = region.area
        if max_region < region.area:
            max_region = region.area
        list_area.append(region.area)
        list_perimeter.append(region.perimeter)
    get_average_perimeter = get_average_perimeter / len(regionprops(label_image))

    for region in regionprops(label_image):
        gray = region.image  # mask_0_1[region.]

        number_pos_px = region.filled_area  # np.count_nonzero(gray
        # print(region.perimeter)
        # print((average_pixel_size*3))
        # print(number_pos_px*0.95)

        # if (number_pos_px*0.95) > (average_pixel_size*2):
        out = ndi.distance_transform_edt(gray)
        out = 1 - out / np.max(out)
        local_maxi = morphology.h_minima(out, 0.05)  # morphology.local_minima(out)
        markers = ndi.label(local_maxi)[0]
        labels = morphology.watershed(out, markers, mask=gray)
        minr, minc, maxr, maxc = region.bbox
        labeled_mask[minr:maxr, minc:maxc] = labels

    plt.imshow(labeled_mask)
    plt.show()

    label_image = label(labeled_mask)
    image_label_overlay = label2rgb(label_image, image=img)
    plt.imshow(image_label_overlay)
    plt.show()

    # Apply C
    heatmap = heatmap_tmp
    import scipy

    batch = []
    coordinations_storage = []

    heatmap_tmp_final = np.zeros((img_org.shape[0], img_org.shape[1], self.nb_class), dtype='float')

    # heatmap_tmp_final[:,:,0] = 1

    # print(object_locations)

    def GenerateRect(slice_x, max_length):
        y_range = slice_x.stop - slice_x.start
        diff = 64 - y_range
        if diff == 0:
            return (slice_x.start, slice_x.stop)

        if diff < 0:
            return (slice_x.start, slice_x.stop)

        middle = int(math.floor(diff * 0.5))
        new_start = slice_x.start - middle
        start_exeed = False
        if new_start < 0:
            middle = middle + new_start
            new_start = 0
            start_exeed = True

        new_stop = slice_x.stop + middle
        if new_stop > max_length:
            diff = new_stop - max_length
            new_stop = new_stop - diff
            if start_exeed is not True:
                new_start = new_start - diff
                if new_start < 0:
                    new_start = 0
        return (new_start, new_stop)

    # Generate the patch images
    for region in regionprops(label_image):
        minr, minc, maxr, maxc = region.bbox

        b_oc_y = slice(minr, maxr)
        b_oc_x = slice(minc, maxc)
        # print(b_oc)
        # Check the size of the slice
        y_range = maxr - minr
        x_range = maxc - minc
        print("y_range", y_range)
        print("x_range", x_range)
        if (y_range > 150) or (x_range > 150):
            continue

        if (y_range < 3) or (x_range < 3):
            continue
        if y_range < target_size[0]:
            start_point, end_point = GenerateRect(b_oc_y, img.shape[0])
            b_oc_y = slice(start_point, end_point)

        if x_range < self.target_size[1]:
            start_point, end_point = GenerateRect(b_oc_x, img.shape[1])
            b_oc_x = slice(start_point, end_point)
        print("b_oc_y", b_oc_y)
        print("b_oc_x", b_oc_x)

        # Get the patch images
        selected_object = img_org[(b_oc_y, b_oc_x)]
        # if (selected_object.shape[0]==0 or selected_object.shape[1]==0):
        #    print("Error", selected_object.shape)
        height, width, _ = selected_object.shape
        # Determine how to split the image to fit to the target size..

        # print(selected_object)
        if (selected_object.shape[0] <= target_size[0] and selected_object.shape[1] <= target_size[1]):
            img_batch_s = np.zeros((target_size[0], target_size[1], number_of_channel), dtype=K.floatx())
            img_batch_s[0:selected_object.shape[0], 0:selected_object.shape[1], :] = selected_object
            coordinations_storage.append(
                [b_oc_x.start, b_oc_y.start, selected_object.shape[0], selected_object.shape[1]])

            batch.append(img_batch_s)
        else:
            # print("else is fired...")
            h_factor = int(math.ceil(height / target_size[0]))
            w_factor = int(math.ceil(width / target_size[1]))

            # print(h_factor)
            # print(w_factor)
            for x_time in range(w_factor):

                for y_time in range(h_factor):
                    left = x_time * target_size[0]
                    top = y_time * target_size[1]
                    # print("left",left)
                    # print("top",top)

                    right = target_size[0] * (1 + x_time)
                    bottom = target_size[1] * (1 + y_time)
                    # print("right",right)
                    # print("bottom",bottom)
                    if right > width:
                        dff = right - width
                        right = right - dff
                        left = left - dff

                    if bottom > height:
                        dff = bottom - height
                        bottom = bottom - dff
                        top = top - dff
                    off_set_y = b_oc_y.start
                    off_set_x = b_oc_x.start
                    # if (off_set_x+left==0):
                    # print([off_set_x+left, off_set_y+top])
                    img_batch_s = np.zeros((target_size[0], target_size[1], number_of_channel),
                                           dtype=K.floatx())
                    # print(selected_object.shape)
                    img_temp_selected = selected_object[top:bottom, left:right, :]
                    # print(img_temp_selected)
                    img_batch_s[0:img_temp_selected.shape[0], 0:img_temp_selected.shape[1], :] = img_temp_selected
                    coordinations_storage.append(
                        [off_set_x + left, off_set_y + top, img_temp_selected.shape[0], img_temp_selected.shape[1]])
                    batch.append(img_batch_s)

                # plt.imshow(selected_object[left:right,top:bottom])
                # plt.show()
    # Convert to keras format:
    batch_patch_img = np.zeros((len(batch), target_size[0], target_size[1], number_of_channel),
                               dtype=K.floatx())
    for i, img_batch in enumerate(batch):
        batch_patch_img[i] = img_batch
    # Generate the mask
    predicted_batches = model.predict(batch_patch_img, batch_size=batch_size, verbose=1)
    predicted_batches = predicted_batches.reshape(batch_patch_img.shape)
    predicted_normalization = np.zeros((img_org.shape[0], img_org.shape[1]), dtype='float')
    for off_sets, img_batch in zip(coordinations_storage, predicted_batches):
        predicted_image_tmp = np.ones((target_size[0], target_size[1], nb_class), dtype='float')
        predicted_image_tmp = (predicted_image_tmp * img_batch)
        predicted_img_tmp = np.around(predicted_image_tmp, decimals=2)
        # print(predicted_img_tmp.shape)
        # print("off_sets[1]+self.target_size[1]", off_sets[1]+self.target_size[0])
        # print("off_sets[0]+self.target_size[0]", off_sets[0]+self.target_size[1])
        heatmap_tmp_final[off_sets[1]:off_sets[1] + off_sets[2], off_sets[0]:off_sets[0] + off_sets[3]] = np.maximum(
            predicted_img_tmp.copy()[0:off_sets[2], 0:off_sets[3], :],
            heatmap_tmp_final[off_sets[1]:off_sets[1] + off_sets[2], off_sets[0]:off_sets[0] + off_sets[3]])
        predicted_normalization[off_sets[1]:off_sets[1] + off_sets[2], off_sets[0]:off_sets[0] + off_sets[3]] += 1.
    # Show the result:
    # heatmap_tmp_final[np.logical_and(heatmap_tmp_final[:,:,0]>=1, heatmap_tmp_final[:,:,2]==0, heatmap_tmp_final[:,:,1]==0)] = (1,0,0)

    # heatmap_tmp_final[:,:,0]  = heatmap_tmp_final[:,:,0] / predicted_normalization #np.ceil(heatmap_tmp_final)
    # heatmap_tmp_final[:,:,1]  = heatmap_tmp_final[:,:,1] / predicted_normalization
    # heatmap_tmp_final[:,:,2]  = heatmap_tmp_final[:,:,2] / predicted_normalization
    np.save("./heatmap_tmp_final.npy", heatmap_tmp_final)
    # heatmap_tmp_final[:,:,1] = cv2.morphologyEx(heatmap_tmp_final[:,:,1], cv2.MORPH_CLOSE, kernel)
    # heatmap_tmp_final[:,:,1] = cv2.morphologyEx(heatmap_tmp_final[:,:,1], cv2.MORPH_OPEN, kernel)
    plt.imshow(heatmap_tmp_final)
    plt.show()
    plt.imshow(heatmap_tmp_final * img_org)
    plt.show()
    img = heatmap_tmp_final[:, :, 1]
    # img = np.max(np.subtract(img,heatmap_tmp_final[:,:,0]),0)
    # img = np.max(np.subtract(img,heatmap_tmp_final[:,:,2]),0)
    img_8 = np.uint8(np.around(img, decimals=0))
    print(img_8.shape)
    img_org_f = img_org.copy()
    img_org_f[:, :, 0] = img_org_f[:, :, 0] * img_8
    img_org_f[:, :, 1] = img_org_f[:, :, 1] * img_8
    img_org_f[:, :, 2] = img_org_f[:, :, 2] * img_8
    plt.imshow(img_org_f)
    plt.show()
    labeled_array, num_features = scipy.ndimage.label(img_8)
    img_org_f[:, :, 0] = img_org[:, :, 0] * labeled_array
    img_org_f[:, :, 1] = img_org[:, :, 1] * labeled_array
    img_org_f[:, :, 2] = img_org[:, :, 2] * labeled_array
    plt.imshow(img_org_f)
    plt.show()
    cv2.imwrite("./labeled_array.png", labeled_array)

    cv2.imwrite("./test.png", heatmap_tmp_final * 255)

    '''
    detector = cv2.SimpleBlobDetector_create()
    # Detect blobs.
    keypoints = detector.detect(img)
    im_with_keypoints = cv2.drawKeypoints(heatmap, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("Keypoints", im_with_keypoints)
    cv2.waitKey(0)
    '''
    img = np.float32(img)
    corners = cv2.cornerHarris(img, 3, 3, 0.0001)
    heatmap_tmp_final[:, :, 2] = heatmap_tmp_final[:, :, 2] * corners
    plt.imshow(corners)
    plt.show()

    labeled_array, num_features = scipy.ndimage.label(img_8)

    plt.imshow(heatmap * img_org)
    plt.show()
    print("Save the heatmap as png image file...")
    cv2.imwrite(filename_to_save_heatmap, heatmap * 255)
    print("End Case")
