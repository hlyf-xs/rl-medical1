import os
import shutil
import random
import numpy as np
import matplotlib.image as maping

def reNameFile():
    source_root_path = "data/innerLV/image/"
    for filename in os.listdir(source_root_path):
        refilename = filename.split(' ')
        refilename = refilename[0] + refilename[len(refilename) - 1]
        # refilename = refilename.rstrip('.png')

        src = source_root_path + filename
        des = source_root_path + refilename
        os.rename(src, des)


        print(filename)


if __name__ == '__main__':
    reNameFile()


