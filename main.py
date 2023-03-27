#Use pip to install both rawpy and opencv-python
import rawpy
import cv2
import numpy as np
import os

#Example code taken from:
#https://stackoverflow.com/questions/45706127/how-to-open-a-cr2-file-in-python-opencv
#https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/

def sampleload():
    with rawpy.imread("IMG_8700.CR2") as raw:
        rgb = raw.postprocess()  # a numpy RGB array
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # the OpenCV image

    scale_percent = 15  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    cv2.imshow('image',resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

####GPT-3 generated code - untested.
def read_raw_images(start_num, end_num):
    img_list = []
    for i in range(start_num, end_num+1):
        filename = 'IMG_{}.CR2'.format(i)
        if os.path.exists(filename):
            with rawpy.imread(filename) as raw:
                img = raw.raw_image_visible.astype(np.float32)
                img_list.append(img)
    return img_list

def generate_gaussian_pyramid(image, levels):
    pyramid = [image]
    for i in range(levels):
        image = cv2.pyrDown(image)
        pyramid.append(image)
    return pyramid

def generate_laplacian_pyramid(image, levels):
    gaussian_pyramid = generate_gaussian_pyramid(image, levels)
    pyramid = []
    for i in range(levels, 0, -1):
        size = (gaussian_pyramid[i - 1].shape[1], gaussian_pyramid[i - 1].shape[0])
#        print(i)
#        gaussian_extended = cv2.copyMakeBorder(cv2.pyrUp(gaussian_pyramid[i]), 0, 1, 0, 1, cv2.BORDER_DEFAULT)
#        print(gaussian_extended.shape)
#        print(gaussian_pyramid[i-1].shape)
        
        gaussian_extended = cv2.pyrUp(gaussian_pyramid[i],dstsize=size)
        laplacian = cv2.subtract(gaussian_pyramid[i-1], gaussian_extended)
        pyramid.append(laplacian)
    pyramid.append(gaussian_pyramid[0])
    return pyramid[::-1]

def laplacian_focus_stack(images, levels):
    laplacian_pyramids = []
    for image in images:
        laplacian_pyramids.append(generate_laplacian_pyramid(image, levels))
    fused_pyramid = []
    for i in range(levels+1):
        laplacian_list = []
        for laplacian_pyr in laplacian_pyramids:
            laplacian_list.append(laplacian_pyr[i])
        fused = cv2.add(laplacian_list[0]/len(laplacian_list), cv2.sum(laplacian_list[1:])[1]/len(laplacian_list))
        fused_pyramid.append(fused)
    fused_image = fused_pyramid[0]
    for i in range(1, levels+1):
        fused_image = cv2.pyrUp(fused_image)
        fused_image = cv2.add(fused_pyramid[i], fused_image)
    return fused_image
def display():
    image1 = cv2.imread('image1.jpg')
    image2 = cv2.imread('image2.jpg')
    image3 = cv2.imread('image3.jpg')
    images = [image1, image2, image3]
    levels = 5
    fused_image = laplacian_focus_stack(images, levels)
    
    cv2.imshow('Fused Image', fused_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()