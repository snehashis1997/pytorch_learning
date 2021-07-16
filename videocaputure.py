import sys
import argparse

import cv2
print(cv2.__version__)

def extractImages(pathIn, pathOut):
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))    # added this line 
        success,image = vidcap.read()
        print ('Read a new frame: ', success)
        cv2.imwrite( pathOut + "\\frame%d.jpg" % count, image)     # save frame as JPEG file
        count = count + 1

if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--pathIn", help="path to video")
    a.add_argument("--pathOut", help="path to images")
    args = a.parse_args()
    print(args)
    extractImages(args.pathIn, args.pathOut)
    
'''

import cv2
 
cap = cv2.VideoCapture(0) # Read camera video stream
fps = cap.get(cv.CAP_PROP_FPS)   # Get the frame rate of the video or the video stream read by the camera
print(type(fps), fps)  # <class 'float'> 30.0   #
cap.set(cv2.CAP_PROP_POS_FRAMES,50)  #Set the frame number to be obtained

c=1
 
if cap.isOpened(): # Whether to open normally
    rval , frame = vc.read()
else:
    rval = False
 
timeF = 30  # Video frame count interval frequency
 
while rval:   # Loop reading video frames
    rval, frame = vc.read()
    if(c%timeF == 0): # Store operation every timeF frame
        cv2.imwrite('image/'+str(c) + '.jpg',frame) # Store as image
    c = c + 1
    cv2.waitKey(1)

cap.release()

'''
