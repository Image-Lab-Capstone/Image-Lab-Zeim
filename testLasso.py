#https://www.pluralsight.com/guides/importing-image-data-into-numpy-arrays
#https://matplotlib.org/users/installing.html#installing-an-official-release
#https://matplotlib.org/3.1.1/gallery/widgets/lasso_selector_demo_sgskip.html

import cv2
import numpy as np



if __name__ == '__main__':

    im = cv2.imread("Oneimage.jpg")
    # Select ROI
    r = cv2.selectROI(im)

    imCrop = im[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
    # Display cropped image
    cv2.imshow("OneImage", imCrop)
    cv2.waitKey(0)

    fromCenter = False
    r = cv2.selectROI(im, fromCenter)


