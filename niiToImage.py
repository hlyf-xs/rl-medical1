import numpy as np
import os
import nibabel as nib
import imageio

def nii_to_image(niifile):
    filenames = os.listdir(filepath)
    slice_trans = []

    for f in filenames:
        img_path = os.path.join(filepath, f)
        img = nib.load(img_path)
        img_fdata = img.get_fdata()
        fname = f.replace('.nii', '')
        img_f_path = os.path.join(imgfile, fname)

        if not os.path.exists(img_f_path):
            os.mkdir(img_f_path)


        (x, y, z, _) = img.shape
        for i in range(z):
            slice = img_fdata[i, :, :]
            imageio.imwrite(os.path.join(img_f_path, '{}.png'.format(i)), slice)


if __name__ == '__main__':
    filepath = '/media/dl/resource/xjy/xs DRL/rl-medical-master/examples/LandmarkDetection/SingleAgent/data/images/'
    imgfile = '/media/dl/resource/xjy/xs DRL/rl-medical-master/examples/LandmarkDetection/SingleAgent/data/images/'
    nii_to_image(filepath)