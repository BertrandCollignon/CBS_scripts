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
	txtfile = open("/home/pi/Desktop/video_" + datetime.datetime.now().strftime("%y-%m-%d_%H:%M:%S") + "_automated.txt", "w")

# otherwise, we are reading from a video file
else:
	camera = cv2.VideoCapture(args["video"])
        filename, fileext = os.path.splitext(os.path.split(args["video"])[1])
	txtfile = open("/home/pi/Desktop/" + filename + "_automated.txt", "w")


txtfile.write("#Time FlowOut\n")


colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 255, 255), (255, 130, 130), (0, 255, 255), (255, 0, 255)]


# initialize the first frame in the video stream
avg = None
avg2 = None
old_frame = None

polygone = None
history = None

count = np.zeros(shape=(1,2))

start_time = time.time()

# loop over the frames of the video
while True:
	# grab the current frame and initialize the occupied/unoccupied
	# text
	(grabbed, frame) = camera.read()

	# if the frame could not be grabbed, then we have reached the end
	# of the video
	if not grabbed:
		break

        frame = frame[(4*frame.shape[0])/10 + 130 : (6*frame.shape[0])/10 + 130,(4*frame.shape[1])/10 + 100: (6*frame.shape[1])/10 - 100,:]

	# resize the frame, convert it to grayscale, and blur it
	frame = resize_even(frame, width=conf["width_resize"])
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	#gray = cv2.equalizeHist(gray)
	#grayHist = gray.copy()
	gray = cv2.GaussianBlur(gray, (conf["kernel_size"], conf["kernel_size"]), 0)
        

        if polygone is None:
                polygone = np.vstack((np.zeros(shape=(gray.shape[0]/2, gray.shape[1]), dtype = 'uint8'), np.ones(shape=(gray.shape[0]/2, gray.shape[1]), dtype = 'uint8')))
                polygone_neg = 1 - polygone
                continue

        if history is None:
                history = np.zeros(shape=(gray.shape[0],gray.shape[1]))
                continue

        history_temp = np.zeros(shape=(gray.shape[0],gray.shape[1]))

        if avg is None:
		avg = gray.copy().astype("float")
		continue

        # accumulate the weighted average between the current frame and
	# previous frames, then compute the difference between the current
	# frame and running average
        
        frameDelta = cv2.subtract(cv2.convertScaleAbs(avg), gray)

        cv2.accumulateWeighted(gray, avg, 0.01)
        
	thresh = cv2.threshold(frameDelta, conf["delta_thresh"], 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        if old_frame is None:
                old_frame = thresh.copy()
                continue

	(_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frame_diff = cv2.bitwise_and(cv2.bitwise_not(old_frame), thresh)
		
	# loop over the contours
	for c in cnts:
             
		# if the contour is too small, ignore it
		if cv2.contourArea(c) < conf["min_area"]:
			continue
											 
		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
		(x, y, w, h) = cv2.boundingRect(c)

                if polygone[y,x] != polygone[y+h, x+w] :

                        history_temp[y:y+h, x:x+w] += 1
                        
                        if np.sum(history[y:y+h, x:x+w]) > 0:
                                continue
                        else:
                                part1 = np.sum(cv2.bitwise_and(polygone[y:y+h, x:x+w], frame_diff[y:y+h, x:x+w])) / (1.0*np.sum(cv2.bitwise_and(polygone[y:y+h, x:x+w], thresh[y:y+h, x:x+w])))
                                part2 = np.sum(cv2.bitwise_and(polygone_neg[y:y+h, x:x+w], frame_diff[y:y+h, x:x+w])) / (1.0*np.sum(cv2.bitwise_and(polygone_neg[y:y+h, x:x+w], thresh[y:y+h, x:x+w])))
                                
                                if part1 > part2:
                                        color_box = (0,0,255)
                                        count[0,0] += 1
                                        txtfile.write(str(time.time()-start_time) + " " + str(-1) +"\n")
                                elif part2 > part1:
                                        color_box = (0, 255, 255)
                                        count[0,1] += 1
                                        txtfile.write(str(time.time()-start_time) + " " + str(1) +"\n")
                                else:
                                        color_box = (0, 0, 0)

                                cv2.rectangle(frame, (x, y), (x + w, y + h), color_box, 2)

                                history[y:y+h, x:x+w] += 1
                        
        history = cv2.bitwise_and(history, history_temp)

        # show the frame and record if the user presses a key

        cv2.drawContours(frame, cnts, -1, (255,255,255))
        cv2.line(frame, (0, frame.shape[0]/2), (frame.shape[1], frame.shape[0]/2),(0,255,0))
        cv2.putText(frame, "Up: {}".format(int(count[0,1])), (10, frame.shape[0]/2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, "Down: {}".format(int(count[0,0])), (10, frame.shape[0]/2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow("Colony monitoring", frame)
        #cv2.imshow("Background", cv2.convertScaleAbs(avg))
        #cv2.imshow("After", grayHist)
        #cv2.imshow("Before", gray)
        key = cv2.waitKey(1) & 0xFF

        old_frame = cv2.threshold(frameDelta, conf["delta_thresh"], 255, cv2.THRESH_BINARY)[1]
        old_frame = cv2.dilate(thresh, np.ones((1,1),np.uint8), iterations=2)
 
        # if the `q` key is pressed, break from the lop
        if key == ord("q"):
                break

# cleanup the camera and close any open windows
txtfile.write(str(time.time()-start_time) + " " + str(0) +"\n")
txtfile.close()
camera.release()
cv2.destroyAllWindows()
