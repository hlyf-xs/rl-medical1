#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: dataReader.py
# Author: Amir Alansary <amiralansary@gmail.com>

import warnings

warnings.simplefilter("ignore", category=ResourceWarning)

import numpy as np
import SimpleITK as sitk
from PIL import Image
import matplotlib.pyplot as plt
import glob  #查找符合特定规则的文件路径名
from tensorpack import logger
from IPython.core.debugger import set_trace

from PIL import Image


__all__ = ['filesListBrainMRLandmark', 'filesListCardioLandmark', 'filesListLVCardiacMRLandmark','filesListFetalUSLandmark', 'NiftiImage']

######################################################################
## extract points from txt file
def getLandmarksFromTXTFile1(file):
    """
        Extract each landmark point line by line from a text file,
        and return vector containing all landmarks.
    """
    with open(file) as fp:
        landmarks = []
        for i, line in enumerate(fp):
            landmarks.append([float(k) for k in line.split(',')])
        landmarks = np.asarray(landmarks).reshape((-1, 3))
        return landmarks

## extract points from txt file
def getLandmarksFromTXTFile(file):
    """
        Extract each landmark point line by line from a text file,
        and return vector containing all landmarks.
    """
    with open(file) as fp:
        landmarks = []
        for i, line in enumerate(fp):
            landmarks.append([float(k) for k in line.split(',')])
        landmarks = np.asarray(landmarks).reshape((-1, 3))
        return landmarks

def getLandmarksFromVTKFile(file):
    """
    Extract each landmark point line by line from a VTK file, and return vector containing all landmarks.
    For cardiac data landmark indexes:
        0-2 RV insert points
        1 -> RV lateral wall turning point
        3 -> LV lateral wall mid-point
        4 -> apex
        5-> center of the mitral valve

        This is a special function for cardiac landmarks, but the dataset is private.
    """

    with open(file) as fp:
        landmarks = []
        for i, line in enumerate(fp):
            if i == 5:
                landmarks.append([float(k) for k in line.split()])
            elif i == 6:
                landmarks.append([float(k) for k in line.split()])
            elif i > 6:
                landmarks = np.asarray(landmarks).reshape((-1, 3))
                # correct landmark according to image direction
                landmarks[:, [0, 1]] = -landmarks[:, [0, 1]]
                return landmarks

###############################################################################
class filesListBrainMRLandmark(object):
    """
        A class for managing train files for mri brain data

        Attributes:
        files_list: Two or one text files that contain a list of all images and (landmarks)
        returnLandmarks: Return landmarks if task is train or eval (default: True)
    """

    def __init__(self, files_list=None, returnLandmarks=True):
        # check if files_list exists
        print("files_list:", files_list)
        assert files_list, 'There is no file given'

        # read image filenames
        print("file_list:", files_list[0])
        self.image_files = [line.split('\n')[0] for line in open(files_list[0])]
        print("image_file:", self.image_files)

        # read landmark filenames if task is train or eval
        self.returnLandmarks = returnLandmarks
        if self.returnLandmarks:
            self.landmark_files = [line.split('\n')[0] for line in open(files_list[1])]
            assert len(self.image_files) == len(
                self.landmark_files), 'number of image files is not equal to number of landmark files'


    @property
    def num_files(self):
        return len(self.image_files)

    def sample_circular(self, shuffle=False):
        """
            return a random sampled ImageRecord from the list of files
        """
        if shuffle:
            indexes = rng.choice(x, len(x), replace=False)
        else:
            indexes = np.arange(self.num_files)

        while True:
            for idx in indexes:
                sitk_image, image = NiftiImage().decode(self.image_files[idx])

                if self.returnLandmarks:
                    # transform landmarks to image space if they are in physical space
                    landmark_file = self.landmark_files[idx]
                    all_landmarks = getLandmarksFromTXTFile(landmark_file)
                    landmarks = all_landmarks[14]  # landmark index is 13 for ac-point and 14 pc-point

                    # transform landmark from physical to image space if required
                    # landmark = sitk_image.TransformPhysicalPointToContinuousIndex(landmark)
                    landmarks = np.round(landmarks).astype('int')
                else:
                    landmarks = None

                # extract filenames from path
                image_filenames = self.image_files[idx][:-7]
                yield image, landmarks, image_filenames, sitk_image.GetSpacing()

###############################################################################

class filesListLVCardiacMRLandmark(object):
    """
        A class for managing train files for mri cardiac data

        Attributes:
        files_list: Two or one text files that contain a list of all images and (landmarks)
        returnLandmarks: Return landmarks if task is train or eval (default: True)
    """

    def __init__(self, files_list=None, returnLandmarks=True):
        # check if files_list exists
        print("files_list:", files_list)
        assert files_list, 'There is no file given'

        # read image filenames
        print("file_list:", files_list[0])
        self.image_files = [line.split('\n')[0] for line in open(files_list[0])]
        print("image_file:", self.image_files)

        # read landmark filenames if task is train or eval
        self.returnLandmarks = returnLandmarks
        if self.returnLandmarks:
            self.landmark_files = [line.split('\n')[0] for line in open(files_list[1])]
            assert len(self.image_files) == len(
                self.landmark_files), 'number of image files is not equal to number of landmark files'


    @property
    def num_files(self):
        return len(self.image_files)

    def sample_circular(self, shuffle=False):
        """
            return a random sampled ImageRecord from the list of files
        """

        if shuffle:
            indexes = rng.choice(x, len(x), replace=False)
        else:
            indexes = np.arange(self.num_files)
            print("index:", indexes)

        while True:
            for idx in indexes:
                sitk_image, image = Read2DImage().decode(self.image_files[idx])

                # print(image.shape)

                if self.returnLandmarks:
                    # transform landmarks to image space if they are in physical space
                    landmark_file = self.landmark_files[idx]
                    # all_landmarks = getLandmarksFromTXTFile(landmark_file)

                    # landmark index is 13 for ac-point and 14 pc-point
                    # this is pc-point
                    print(landmark_file)
                    x_point = str(landmark_file[0]) + str(landmark_file[1])
                    y_point = str(landmark_file[len(landmark_file)-2]) + str(landmark_file[len(landmark_file)-1])

                    landmarks = [int(x_point), int(y_point)]
                    print("landmark:", landmarks)

                    # transform landmark from physical to image space if required
                    # landmark = sitk_image.TransformPhysicalPointToContinuousIndex(landmark)
                    landmarks = np.round(landmarks).astype('int')
                else:
                    landmarks = None

                # extract filenames from path
                image_filenames = self.image_files[idx][:-7]
                yield image, landmarks, image_filenames, sitk_image

###############################################################################


class filesListCardioLandmark(object):
    """ A class for managing train files for mri cardiac data
        Attributes:
        files_list: Two or one text files that contain a list of all images and (landmarks)
        returnLandmarks: Return landmarks if task is train or eval (default: True)
    """

    def __init__(self, files_list=None, returnLandmarks=True, agents=1):
        # check if files_list exists
        assert files_list, 'There is no file given'

        # read image filenames
        self.image_files = [line.split('\n')[0] for line in open(files_list[0].name)]

        # read landmark filenames if task is train or eval
        self.returnLandmarks = returnLandmarks
        self.agents = agents

        if self.returnLandmarks:
            self.landmark_files = [line.split('\n')[0] for line in open(files_list[1].name)]

            assert len(self.image_files) == len(
                self.landmark_files), 'number of image files is not equal to number of landmark files'

    @property
    def num_files(self):
        return len(self.image_files)

    # 循环采样？
    def sample_circular(self, shuffle=False):
        """
            return a random sampled ImageRecord from the list of files
        """
        if shuffle:
            indexes = rng.choice(x, len(x), replace=False)
        else:
            indexes = np.arange(self.num_files)

        while True:
            for idx in indexes:
                sitk_image, image = NiftiImage().decode(self.image_files[idx])

                if self.returnLandmarks:
                    landmark_file = self.landmark_files[idx]
                    all_landmarks = getLandmarksFromVTKFile(landmark_file)

                    # transform landmarks to image coordinates
                    all_landmarks = [sitk_image.TransformPhysicalPointToContinuousIndex(point)
                                     for point in all_landmarks]

                    # Indexes: 0-2 RV insert points, 1 -> RV lateral wall turning point, 3 -> LV lateral wall mid-point,
                    # 4 -> apex, 5-> center of the mitral valve
                    landmarks = [np.round(all_landmarks[(i + 4) % 6]) for i in range(self.agents)]  # Apex + MV

                    # landmarks = [np.round(all_landmarks[(i + 3) % 6]) for i in range(self.agents)]  # LV + Apex
                    # landmarks = [np.round(all_landmarks[((i + 1) + 3) % 6]) for i in range(self.agents)] # LV + MV
                else:
                    landmarks = None

                # extract filenames from path, remove .nii.gz extension
                image_filenames = [self.image_files[idx][:-7]] * self.agents
                images = [image] * self.agents

                yield images, landmarks, image_filenames, sitk_image.GetSpacing()

###############################################################################

class filesListFetalUSLandmark(object):
    """
        A class for managing train files for fetal ultrasound data
        Attributes:
        files_list: Two or one text files that contain a list of all images and (landmarks)
        returnLandmarks: Return landmarks if task is train or eval (default: True)
    """
    def __init__(self, files_list=None, returnLandmarks=True, agents=1):
        # check if files_list exists
        assert files_list, 'There is no file given'
        # read image filenames
        self.image_files = [line.split('\n')[0] for line in open(files_list[0].name)]
        # read landmark filenames if task is train or eval
        self.returnLandmarks = returnLandmarks
        self.agents = agents
        if self.returnLandmarks:
            self.landmark_files = [line.split('\n')[0] for line in open(files_list[1].name)]
            assert len(self.image_files) == len(
                self.landmark_files), 'number of image files is not equal to number of landmark files'

    @property
    def num_files(self):
        return len(self.image_files)

    def sample_circular(self, shuffle=False):
        """ return a random sampled ImageRecord from the list of files
        """
        if shuffle:
            indexes = rng.choice(x, len(x), replace=False)
        else:
            indexes = np.arange(self.num_files)

        while True:
            for idx in indexes:
                sitk_image, image = NiftiImage().decode(self.image_files[idx])

                if self.returnLandmarks:
                    landmark_file = self.landmark_files[idx]
                    all_landmarks = getLandmarksFromVTKFile(landmark_file)
                    # landmark point 12 csp - 11 leftCerebellar - 10 rightCerebellar
                    # landmarks = [np.round(all_landmarks[(i*2 + 10) % 13]) for i in range(self.agents)]
                    landmarks = [np.round(all_landmarks[(i + 10) % 13]) for i in range(self.agents)]  # Apex + MV

                else:
                    landmarks = None

                # extract filenames from path, remove .nii.gz extension
                image_filenames = [self.image_files[idx][:-7]] * self.agents
                images = [image] * self.agents

                yield images, landmarks, image_filenames, sitk_image.GetSpacing()

###############################################################################

class ImageRecord(object):
    '''
        image object to contain height, width, depth and name
    '''
    pass

#
class NiftiImage(object):
    """
        Helper class that provides TensorFlow image coding utilities.
    """
    def __init__(self):
        pass

    def _is_nifti(self, filename):
        """
            Determine if a file contains a nifti format image.
            Args
              filenames: string, path of the image file
            Returns
              boolean indicating if the image is a nifti
        """
        extensions = ['.nii', '.nii.gz', '.img', '.hdr']
        return any(i in filename for i in extensions)

    def decode(self, filename, label=False):
        """
            decode a single nifti image
            Args
              filenames: string for input images
              label: True if nifti image is label
            Returns
              image: an image container with attributes; name, data, dims
        """

        image = ImageRecord()
        image.name = filename
        assert self._is_nifti(image.name), "unknown image format for %r" % image.name

        if label:
            sitk_image = sitk.ReadImage(image.name, sitk.sitkInt8)
        else:
            sitk_image = sitk.ReadImage(image.name, sitk.sitkFloat32)
            np_image = sitk.GetArrayFromImage(sitk_image)

            # threshold image between p10 and p98 then re-scale [0-255]
            p0 = np_image.min().astype('float')
            p10 = np.percentile(np_image, 10)
            p99 = np.percentile(np_image, 99)
            p100 = np_image.max().astype('float')

            # logger.info('p0 {} , p5 {} , p10 {} , p90 {} , p98 {} , p100 {}'.format(p0, p5, p10, p90, p98, p100))
            sitk_image = sitk.Threshold(sitk_image,
                                        lower=p10,
                                        upper=p100,
                                        outsideValue=p10)

            sitk_image = sitk.Threshold(sitk_image,
                                        lower=p0,
                                        upper=p99,
                                        outsideValue=p99)

            sitk_image = sitk.RescaleIntensity(sitk_image,
                                               outputMinimum=0,
                                               outputMaximum=255)

        # Convert from [depth, width, height] to [width, height, depth]
        image.data = sitk.GetArrayFromImage(sitk_image).transpose(2, 1, 0) #.astype('uint8')
        image.dims = np.shape(image.data)

        return sitk_image, image

class Read2DImage(object):
    def __init__(self):
        pass

    def _is_image(self, filename):
        """
            Determine if a file contains a 2D format image.
            Args
              filenames: string, path of the image file
            Returns
              boolean indicating if the image is a nifti
        """
        extensions = ['.png', '.jpg']
        return any(i in filename for i in extensions)

    def decode(self, filename, label=False):
        """
            decode a single nifti image
            Args
              filenames: string for input images
              label: True if nifti image is label
            Returns
              image: an image container with attributes; name, data, dims
        """

        image = ImageRecord()
        image.name = filename

        print("image.name:", image.name)

        assert self._is_image(image.name), "unknown image format for %r" % image.name

        image = Image.open(image.name)  # 读取图片文件

        image_arr = np.array(image)  # 将图片以数组的形式读入变量
        print(image_arr)

        imageSpace = (1.0, 1.0)

        # Convert from [depth, width, height] to [width, height, depth]
        image.dims = np.shape(image)

        return imageSpace, image