#Video to image conversion function from https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames
import cv2

vc = cv2.VideoCapture('checkerboard_000.h264') # or ('checkerboard_019.h264')
c=0

if vc.isOpened():
    rval , frame = vc.read()
else:
    rval = False

while rval:
    rval, frame = vc.read()
    cv2.imwrite(str(c) + '.jpg',frame)
    c = c + 1
    cv2.waitKey(1)
vc.release()
