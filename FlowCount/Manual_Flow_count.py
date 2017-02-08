import argparse
import datetime
import json
import time
import cv2
import numpy as np
import random
import os


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-c", "--conf", help="path to the config file")
args = vars(ap.parse_args())

conf = json.load(open(args["conf"]))



def resize_even(image, width = None, height = None, inter = cv2.INTER_AREA):

	dim = None
	(h, w) = image.shape[:2]

	if width is None and height is None:
		return image

	if width is None:
		r = height / float(h)
		dim = (2*(int((w * r)/2)), height)

	else:
		r = width / float(w)
		dim = (width, 2*(int((h * r)/2)))

	resized = cv2.resize(image, dim, interpolation = inter)

	return resized


# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
	camera = cv2.VideoCapture(1)
	time.sleep(0.25)
        txtfile = open("/home/pi/Desktop/video_" + datetime.datetime.now().strftime("%y-%m-%d_%H:%M:%S") + "_manual.txt", "w")

# otherwise, we are reading from a video file
else:
	camera = cv2.VideoCapture(args["video"])
	filename, fileext = os.path.splitext(os.path.split(args["video"])[1])
	txtfile = open("/home/pi/Desktop/" + filename + "_manual.txt", "w")


txtfile.write("#Time FlowOut\n")

count = np.zeros(shape=(1,2))

start_time = time.time()
pause_time = 0


# loop over the frames of the video
while True:
	# grab the current frame and initialize the occupied/unoccupied
	# text
	(grabbed, frame) = camera.read()

	# if the frame could not be grabbed, then we have reached the end
	# of the video
	if not grabbed:
		break

        frame = frame[ 75 : frame.shape[0] - 40 , 250 : frame.shape[1]-750, : ]

	# resize the frame, convert it to grayscale, and blur it
	#frame = resize_even(frame, width=conf["width_resize"])

        cv2.line(frame, (frame.shape[1]/2, 0), (frame.shape[1]/2, frame.shape[0]),(0,255,0))
        cv2.putText(frame, "Up: {}".format(int(count[0,1])), (10, frame.shape[0]/2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, "Down: {}".format(int(count[0,0])), (10, frame.shape[0]/2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow("Colony monitoring", frame)

        key = cv2.waitKey(1) & 0xFF

        # if the `q` key is pressed, break from the lop
        
        if key == 82:
                count[0,1] += 1
                txtfile.write(str(time.time()-start_time - pause_time) + " " + str(1) +"\n")
        elif key == 84:
                count[0,0] += 1
                txtfile.write(str(time.time()-start_time - pause_time) + " " + str(-1) +"\n")
        elif key == ord("p"):
                start_pause = time.time()
                cv2.putText(frame,"PAUSE" ,(frame.shape[1]/2, frame.shape[0]/2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.imshow("Colony monitoring", frame)
                cv2.waitKey(0) & 0xFF
                stop_pause = time.time()
                pause_time += stop_pause - start_pause
        elif key == ord("q"):
                break

# cleanup the camera and close any open windows
txtfile.write(str(time.time()-start_time - pause_time) + " " + str(0) +"\n")
txtfile.close()
camera.release()
cv2.destroyAllWindows()
