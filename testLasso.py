#https://www.pluralsight.com/guides/importing-image-data-into-numpy-arrays
#https://matplotlib.org/users/installing.html#installing-an-official-release
#https://matplotlib.org/3.1.1/gallery/widgets/lasso_selector_demo_sgskip.html
#https://www.learnopencv.com/how-to-select-a-bounding-box-roi-in-opencv-cpp-python/

import cv2
import numpy as np
import pdb


def decrease_intesity(im, r, value=10):
    '''https://www.youtube.com/watch?v=WxKS6Uo5n8c
    Decreases image pixel intesity for pixels outside of bounding box
    '''
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            for k in range(im.shape[2]):
                #check if current pixel location is within bounding box
                if(not(i>int(r[1]) and i < int(r[1]+r[3]) and j > int(r[0]) and j < int(r[0]+r[2]))):
                    if((im[i,j,k] - value) < 0):
                        im[i,j,k] = 0
                    elif((im[i,j,k] - value) > 255):
                        im[i,j,k] = 255
                    else:
                        im[i,j,k] = im[i,j,k] - value
    return im

def select_roi(im, val=50):
    r = cv2.selectROI(im)
    print(r)
    new_im = decrease_intesity(im, r, value=val)
    return new_im




