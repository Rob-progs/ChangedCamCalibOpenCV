# Changed Camera Calibration code from https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html
import numpy as np
import cv2 as cv
import glob
import os
from checkerboard import detect_checkerboard

################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

chessboardSize = (6,6)
frameSize = (1920,1080)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)


# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

image_read_counter = 0
images = glob.glob('*.jpg')

# Loop over all images in the directory of the code
for image in images:

    # Increment and print counter for maintainance purpose
    image_read_counter = image_read_counter + 1
    print("Reading Image:", image_read_counter)

    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners -> original OpenCV function
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    # Function to determine the quality of the chessboard detection  ( https://pypi.org/project/checkerboard/ )
    cornerpoints, score = detect_checkerboard(gray, chessboardSize)
    # Important to convert float64 -> float32 when using this function
    # corners = np.float32(corners)




    # If a chessboard is found and the detection has a good quality, add object points and image points after refining them
    if (ret == True) and (score < 0.5):

        # Copy valid images to ValidImages folder... For organisational purpose
        #cv.imwrite('/ValidImages/' + str(
        #    image_read_counter) + '.jpg', img)

        print("Processing Image:", image_read_counter)

        #cv.waitKey(0)

        objpoints.append(objp)

        # termination criteria for refinement
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # Refining pixel positions
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2) # Here the OpenCV code has a Error and appends unrefined points

        # Draw and display the corners
        #cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        #cv.imshow('img', img)
        #cv.imwrite('Pattern' + str(image_read_counter) + '.png',img)
        #cv.waitKey(2)


cv.destroyAllWindows()


############## CALIBRATION #######################################################

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
print('RMS calibration error:', ret, '\n', '\ncameraMatrix:\n', cameraMatrix, '\n', '\nDistorsion:', dist, '\n\nrvecs:\n', rvecs, '\n\ntvecs:\n', tvecs)

############## UNDISTORTION #####################################################
#Image array for images to be undistorted
img = []

#Images of first video
img1 = cv.imread('/CalibVideo1/163.jpg')
img.append(img1)
img2 = cv.imread('/CalibVideo1/269.jpg')
img.append(img2)
img3 = cv.imread('/CalibVideo1/741.jpg')
img.append(img3)
img4 = cv.imread('/CalibVideo1/695.jpg')
img.append(img4)


#Images of second video
img5 = cv.imread('/CalibVideo2/1.jpg')
img.append(img5)
img6 = cv.imread('/CalibVideo2/645.jpg')
img.append(img6)
img7 = cv.imread('/CalibVideo2/1682.jpg')
img.append(img7)
img8 = cv.imread('/CalibVideo2/1746.jpg')
img.append(img8)

write_counter = 1

cameraMatrix = np.zeros(shape=(3, 3), dtype=float)
newCameraMatrix = np.zeros(shape=(3, 3), dtype=float)
dist = np.zeros(shape=(0, 3), dtype=float)

# Determined parameters and distortions
#cameraMatrix =  np.array([[1308.48240, 0.00000000, 943.249037],[0.00000000, 1312.24559, 462.244300],[0.00000000, 0.00000000, 1.00000000]])
#newCameraMatrix = np.array([[784.14855957, 0., 873.06276642], [0., 804.65039062, 404.46434741], [0., 0., 1.]])
#dist = np.array([[-3.66140985e-01,  1.88285247e-01,  7.40083619e-04, -1.42772163e-04, -5.97380717e-02]])


# Loop to undistord images
for imag in img:
    h,  w = img1.shape[:2]
    newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))
    print('\nnewCameraMatrix:\n', newCameraMatrix)

    # Undistortion method 1
    dst = cv.undistort(imag, cameraMatrix, dist, None, newCameraMatrix)

    # Save undistorted images with black spots
    #cv.imwrite('caliResultdist' + str(write_counter) + '.png', dst)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    #store cropped image
    cv.imwrite('caliResult' + str(write_counter) + '.png', dst)

    write_counter = write_counter + 1



############## CALCULATING THE REPROJECTION ERROR #####################################################
mean_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

print( "\ntotal error: {}".format(mean_error/len(objpoints)) )
