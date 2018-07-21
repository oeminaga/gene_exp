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
            image = OpenSlide(filename)
            # temp = str(image.properties)[14:-1].replace("u'", "\"")
            # temp = temp.replace("'", r'"')
            # temp = json.loads(temp)
        except OSError:
            print(OSError.message)
        return image

    def LoadMask(self, filename, return_identity_list=False):
        LoadAll = False
        if self.select_criteria is None:
            LoadAll = True

        polygons = self.ExtractPolygons(ET.parse(filename).getroot(), return_identity_list=return_identity_list, LoadAll=LoadAll, select_criteria=self.select_criteria)
        return polygons

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

    # Begin: Major functions (Also an example use for other functions listed above)
    def CreatePatchesToStoreAsFiles(self, filename, annotation_file):
        '''
        :param filename: the file path for the histology images supported by OpenSlide
        :param annotation_file: the annotation file path: It should be a XML file stored in a ImageScope format
        :param Debug: Not active
        :return: A log_report in numpy format.
        '''
        mask_dir_to_store = self.image_folder
        img_dir_to_store = self.mask_folder

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

    def LoadFiles(self):
        mask_files = self.GenerateFileListFromDirectory(self.mask_directories,self.mask_extension)
        img_files = self.GenerateFileListFromDirectory(self.image_directories, self.img_extension)
        self.files = self.FindMatchedImagesAndAnnotation(img_files, mask_files)

