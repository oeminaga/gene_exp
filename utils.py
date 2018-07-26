#####
#Copyright by Okyaz Eminaga, 2017-2018
#This is part includes frequently used functions.
#####

import os
import numpy as np
import cv2
from PIL import Image
import xml.etree.ElementTree as ET
from openslide import OpenSlide
import keras.backend as K
import random
from skimage import color, filters, morphology
from skimage.morphology import square
import skimage
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border


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
        self.factor_for_mask = self.target_zooming / self.annotation_zooming_level
        self.factor = self.target_zooming / self.zooming_level
        self.target_zooming = target_zooming

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
        print(filename)
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
        image = self.image.read_region((0, 0), level, self.img.level_dimensions[level])
        image = np.asarray(image)
        image = image[:, :, 0:3]
        HE = color.rgb2hed(image)
        BackgroundColor_0 = np.max(HE[0:10, 0:10, 2])
        tissues = HE[:, :, 2]
        tissues = filters.gaussian(tissues, 2)
        tissues = tissues >= (BackgroundColor_0 * 0.99)
        tissues = morphology.opening(tissues, square(10))
        return tissues

    def LoadTissueMask(self):
        level_3 = self._GettissueArea(3)
        level_2 = self._GettissueArea(2)

        shape_To_convert = (self.image.level_dimensions[2][1], self.image.level_dimensions[2][0])
        level_3_u = skimage.transform.resize(level_3, shape_To_convert)
        level = level_3_u * level_2
        cleared = clear_border(level)
        label_image = label(cleared)
        regions = regionprops(label_image)

        level_factor = np.divide(self.image.level_dimensions[0], self.image.level_dimensions[2])
        minr, minc, maxr, maxc = regions[0].bbox
        minr_new = minr * level_factor
        minc_new = minc * level_factor

        maxr_new = maxr * level_factor
        maxc_new = maxc * level_factor

        rect = (np.around((minr_new[1], minc_new[0], maxr_new[1] - minr_new[1], maxc_new[0] - minc_new[0]),
                          decimals=0)).astype(np.int)

        level_0 = skimage.transform.resize(regions[0].filled_image, shape_To_convert)
        return level_0,  rect

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

    def RandomRegionDefinition(self, mask, max_patch_number, patch_size):
        '''
        :param mask: mask image
        :param max_patch_number: Define the batch size
        :param patch_size: Define the patch dimension
        :return: A list of X,Y coordinations randomly defined.
        '''
        dimension = mask.shape
        reg_lst = []
        counter = 0
        total_size = patch_size[0] * patch_size[1]
        while counter < max_patch_number:
            x = random.randint(0, dimension[0] - patch_size[0])
            y = random.randint(0, dimension[1] - patch_size[0])
            mask_selected = mask[y:y + patch_size[1], x:x + patch_size[0]]
            number_positive = np.count_nonzero(mask_selected)
            percentage_positive = number_positive / total_size
            if percentage_positive > 0.99:
                reg_lst.append([x, y])
                counter = counter + 1

        return reg_lst

    def GetPatches(self, image, mask, batch_size, patch_size):
        '''
        :param image: the original image
        :param mask:  the mask (2-Dimension boolean)
        :param batch_size: Define the batch size for this image 100
        :param patch_size: the dimension of the patch (512, 512)
        :return: patched images in numpy array (batch_size, height, width, channels.
        '''
        regions = self.RandomRegionDefinition(mask, batch_size, patch_size)
        patch_imgs = np.zeros((batch_size, patch_size[0], patch_size[1], image.shape[2]), dtype=np.uint8)

        for index, region in enumerate(regions):
            x, y = region
            patch_imgs[index] = image[y:y + patch_size[1], x:x + patch_size[0]].copy()
        return patch_imgs

    def GetPatchesAsFiles(self, image,mask, patch_per_image, patch_size, filename, type_data="train", n_class=""):
        regions = self.RandomRegionDefinition(mask, patch_per_image, patch_size)
        file_ex = os.path.basename(filename)
        file_to_use = os.path.splitext(file_ex)[0]

        x_file_path = self.image_folder + "/" + type_data + "/"+ n_class + "/"
        counter = 0
        for index, region in enumerate(regions):
            x, y = region
            x_file_path = x_file_path + "%s_%s.png" %(file_to_use, counter)
            img = image[y:y + patch_size[1], x:x + patch_size[0]].copy()
            cv2.imwrite(x_file_path, img)


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

    def GeneratePatchDirect(self, filename, patch_size=(512,512), batch_size=100):
        self.LoadImage(filename)
        level_0, region_def = self.LoadTissueMask()
        image = self.image.read_region((region_def[1], region_def[0]), level=0, size=(region_def[3], region_def[2]))
        patch_images = self.GetPatch(image, level_0, batch_size, patch_size)
        return patch_images

    def GeneratePatchDirectAsImagePatch(self, filename, patch_size=(512,512), batch_size=100, type_data="", n_class=""):
        self.LoadImage(filename)
        level_0, region_def = self.LoadTissueMask()
        image = self.image.read_region((region_def[1], region_def[0]), level=0, size=(region_def[3], region_def[2]))
        self.GetPatchesAsFiles(image, level_0, batch_size, patch_size, filename, type_data, n_class)
    # End Major functions

class Filedirectoryamagement():
    def __int__(self, mask_directories, image_directories, image_extension=[".svs", ".tiff"], mask_extension=[".xml"]):
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
        for dir_c in directories:
            if (os.path.exists(dir_c)):
                for (dirpath, dirnames, filenames) in os.walk(dir_c):
                    for filename in filenames:
                        # print(extension)
                        if filename.endswith(tuple(extension)):
                            list_of_files[filename] = os.sep.join([dirpath, filename])
        return list_of_files

    def LoadFiles(self, Mask=True):
        img_files = self.GenerateFileListFromDirectory(self.image_directories, self.img_extension)
        if Mask:
            mask_files = self.GenerateFileListFromDirectory(self.mask_directories,self.mask_extension)
            self.files = self.FindMatchedImagesAndAnnotation(img_files, mask_files)
        else:
            self.files = img_files
            self.files = random.shuffle(self.files)

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

    def generate_patch_images_train_valid_test(self, patch_per_image=100 , image_patch_size=(512,512), subset_ratio=[0.6,0.8,0.2], directory="", class_filename="", type_class_col="sample_type"):
        train_length = int(round(len(self.files[0]) * subset_ratio[0]))
        remaining = len(self.files[0]) - train_length
        test_length = int(round(remaining * subset_ratio[1]))
        valid_length = remaining - test_length

        import pandas as pd
        data = pd.read_csv(class_filename, index_col=0)

        train_files = random.sample(self.files[0], train_length)
        remaining = [e for e in self.files[0] if e not in train_files]
        test_files = random.sample(remaining, test_length)
        valid_files =  [e for e in remaining if e not in test_files]

        files_subsection = [train_files, test_files, valid_files]
        bdx = OpenSlideOnlivePatch(image_folder=directory)
        #Sample id...
        for (filelist) in files_subsection:
            for filename in filelist:
                str_file = os.path.basename(filename)
                str_file = os.path.splitext(str_file)[0]
                sample_id = str_file[0:15]
                data.loc[sample_id, type_class_col]

        for (sample_id, sample_type) in zip(sample_id_list, sample_type_list):
            sampleid = sample_id[0:15]
            filename = self.GetFilename(sample_id)
            bdx.GeneratePatchDirectAsImagePatch(filename, image_patch_size, patch_per_image,)

        bdx.GetPatchesAsFiles()
        for i in range(0, len(self.files), batch_file_size):
            count_files = i + batch_file_size
            if count_files > len(self.files):
                count_files = len(self.files)
            self.batch_files = self.files[i:count_files]
            np.zeros((img_patch_from_each_case,))



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





