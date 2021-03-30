import matplotlib
from matplotlib import pylab as plt
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D

file = '/media/dl/resource/xjy/xs DRL/rl-medical-master/examples/LandmarkDetection/SingleAgent/data/images/' \
       'ADNI_002_S_0816_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070217005829488_S18402_I40731_Normalized_to_002_S_0295.nii.gz'  # 你的nii或者nii.gz文件路径
img = nib.load(file)

print(img)
print(img.header['db_name'])  # 输出nii的头文件

width, height, queue = img.dataobj.shape

OrthoSlicer3D(img.dataobj).show()

num = 1
for i in range(0, queue, 10):
    img_arr = img.dataobj[:, :, i]
    plt.subplot(5, 4, num)
    plt.imshow(img_arr, cmap='gray')
    num += 1

plt.show()