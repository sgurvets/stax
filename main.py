#Use pip to install both rawpy and opencv-python
import rawpy
import cv2

#Example code taken from:
#https://stackoverflow.com/questions/45706127/how-to-open-a-cr2-file-in-python-opencv
#https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/

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