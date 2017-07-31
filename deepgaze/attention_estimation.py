# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import pygame 
import argparse
import imutils
import time
import dlib
import cv2
import math
import sys
from deepgaze.haar_cascade import haarCascade
from deepgaze.face_landmark_detection import faceLandmarkDetection

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style


def sound_alarm(path):
	# play an alarm sound
	pygame.mixer.init()
	pygame.mixer.music.load(path)
	pygame.mixer.music.play()

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear
 

######## Head Pose Part Code #########
#
#If True enables the verbose mode
DEBUG = True 

#Antropometric constant values of the human head. 
#Found on wikipedia and on:
# "Head-and-Face Anthropometric Survey of U.S. Respirator Users"
#
#X-Y-Z with X pointing forward and Y on the left.
#The X-Y-Z coordinates used are like the standard
# coordinates of ROS (robotic operative system)
P3D_RIGHT_SIDE = np.float32([-100.0, -77.5, -5.0]) #0
P3D_GONION_RIGHT = np.float32([-110.0, -77.5, -85.0]) #4
P3D_MENTON = np.float32([0.0, 0.0, -122.7]) #8
P3D_GONION_LEFT = np.float32([-110.0, 77.5, -85.0]) #12
P3D_LEFT_SIDE = np.float32([-100.0, 77.5, -5.0]) #16
P3D_FRONTAL_BREADTH_RIGHT = np.float32([-20.0, -56.1, 10.0]) #17
P3D_FRONTAL_BREADTH_LEFT = np.float32([-20.0, 56.1, 10.0]) #26
P3D_SELLION = np.float32([0.0, 0.0, 0.0]) #27
P3D_NOSE = np.float32([21.1, 0.0, -48.0]) #30
P3D_SUB_NOSE = np.float32([5.0, 0.0, -52.0]) #33
P3D_RIGHT_EYE = np.float32([-20.0, -65.5,-5.0]) #36
P3D_RIGHT_TEAR = np.float32([-10.0, -40.5,-5.0]) #39
P3D_LEFT_TEAR = np.float32([-10.0, 40.5,-5.0]) #42
P3D_LEFT_EYE = np.float32([-20.0, 65.5,-5.0]) #45
#P3D_LIP_RIGHT = np.float32([-20.0, 65.5,-5.0]) #48
#P3D_LIP_LEFT = np.float32([-20.0, 65.5,-5.0]) #54
P3D_STOMION = np.float32([10.0, 0.0, -75.0]) #62

#The points to track
#These points are the ones used by PnP
# to estimate the 3D pose of the face
TRACKED_POINTS = (0, 4, 8, 12, 16, 17, 26, 27, 30, 33, 36, 39, 42, 45, 62)
ALL_POINTS = list(range(0,68)) #Used for debug only


def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotationMatrixToEulerAngles(R) :

    assert(isRotationMatrix(R))
    
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    
    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])



def headPose():

    #Defining the video capture object
    video_capture = cv2.VideoCapture(0)

    if(video_capture.isOpened() == False):
        print("Error: the resource is busy or unvailable")
    else:
        print("The video source has been opened correctly...")

    #Create the main window and move it
    cv2.namedWindow('Video')
    cv2.moveWindow('Video', 20, 20)

    #Obtaining the CAM dimension
    cam_w = int(video_capture.get(3))
    cam_h = int(video_capture.get(4))

    #Defining the camera matrix.
    #To have better result it is necessary to find the focal
    # lenght of the camera. fx/fy are the focal lengths (in pixels) 
    # and cx/cy are the optical centres. These values can be obtained 
    # roughly by approximation, for example in a 640x480 camera:
    # cx = 640/2 = 320
    # cy = 480/2 = 240
    # fx = fy = cx/tan(60/2 * pi / 180) = 554.26
    c_x = cam_w / 2
    c_y = cam_h / 2
    f_x = c_x / np.tan(60/2 * np.pi / 180)
    f_y = f_x

    #Estimated camera matrix values.
    camera_matrix = np.float32([[f_x, 0.0, c_x],
                                   [0.0, f_y, c_y], 
                                   [0.0, 0.0, 1.0] ])

    print("Estimated camera matrix: \n" + str(camera_matrix) + "\n")

    #These are the camera matrix values estimated on my webcam with
    # the calibration code (see: src/calibration):
    # camera_matrix = np.float32([[602.10618226,          0.0, 320.27333589],
    #                                [         0.0, 603.55869786,  229.7537026], 
    #                                [         0.0,          0.0,          1.0] ])

    #Distortion coefficients
    camera_distortion = np.float32([0.0, 0.0, 0.0, 0.0, 0.0])

    #Distortion coefficients estimated by calibration
    #camera_distortion = np.float32([ 0.06232237, -0.41559805,  0.00125389, -0.00402566,  0.04879263])


    #This matrix contains the 3D points of the
    # 11 landmarks we want to find. It has been
    # obtained from antrophometric measurement
    # on the human head.
    landmarks_3D = np.float32([P3D_RIGHT_SIDE,
                                  P3D_GONION_RIGHT,
                                  P3D_MENTON,
                                  P3D_GONION_LEFT,
                                  P3D_LEFT_SIDE,
                                  P3D_FRONTAL_BREADTH_RIGHT,
                                  P3D_FRONTAL_BREADTH_LEFT,
                                  P3D_SELLION,
                                  P3D_NOSE,
                                  P3D_SUB_NOSE,
                                  P3D_RIGHT_EYE,
                                  P3D_RIGHT_TEAR,
                                  P3D_LEFT_TEAR,
                                  P3D_LEFT_EYE,
                                  P3D_STOMION])

    #Declaring the two classifiers
    my_cascade = haarCascade("/home/ashfak/Desktop/DriversAssistanceSystem/deepgaze/etc/xml/haarcascade_frontalface_alt.xml", "/home/ashfak/Desktop/DriversAssistanceSystem/deepgaze/etc/xml/haarcascade_profileface.xml")
    #TODO If missing, example file can be retrieved from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    my_detector = faceLandmarkDetection('/home/ashfak/Desktop/DriversAssistanceSystem/deepgaze/etc/shape_predictor_68_face_landmarks.dat')

    #Error counter definition
    no_face_counter = 0

    #Variables that identify the face
    #position in the main frame.
    face_x1 = 0
    face_y1 = 0
    face_x2 = 0
    face_y2 = 0
    face_w = 0
    face_h = 0

    #Variables that identify the ROI
    #position in the main frame.
    roi_x1 = 0
    roi_y1 = 0
    roi_x2 = cam_w
    roi_y2 = cam_h
    roi_w = cam_w
    roi_h = cam_h
    roi_resize_w = int(cam_w/10)
    roi_resize_h = int(cam_h/10)

    while(True):

        # Capture frame-by-frame
        ret, framePose = video_capture.read()
        framePose = imutils.resize(framePose, width=800)
        gray2 = cv2.cvtColor(framePose[roi_y1:roi_y2, roi_x1:roi_x2], cv2.COLOR_BGR2GRAY)


        #Looking for faces with cascade
        #The classifier moves over the ROI
        #starting from a minimum dimension and augmentig
        #slightly based on the scale factor parameter.
        #The scale factor for the frontal face is 1.10 (10%)
        #Scale factor: 1.15=15%,1.25=25% ...ecc
        #Higher scale factors means faster classification
        #but lower accuracy.
        #
        #Return code: 1=Frontal, 2=FrontRotLeft, 
        # 3=FrontRotRight, 4=ProfileLeft, 5=ProfileRight.
        my_cascade.findFace(gray2, True, True, True, True, 1.10, 1.10, 1.15, 1.15, 40, 40, rotationAngleCCW=30, rotationAngleCW=-30, lastFaceType=my_cascade.face_type)

        #Accumulate error values in a counter
        if(my_cascade.face_type == 0): 
            no_face_counter += 1

        #If any face is found for a certain
        #number of cycles, then the ROI is reset
        if(no_face_counter == 50):
            no_face_counter = 0
            roi_x1 = 0
            roi_y1 = 0
            roi_x2 = cam_w
            roi_y2 = cam_h
            roi_w = cam_w
            roi_h = cam_h

        #Checking wich kind of face it is returned
        if(my_cascade.face_type > 0):

            #Face found, reset the error counter
            no_face_counter = 0

            #Because the dlib landmark detector wants a precise
            #boundary box of the face, it is necessary to resize
            #the box returned by the OpenCV haar detector.
            #Adjusting the framePose for profile left
            if(my_cascade.face_type == 4):
                face_margin_x1 = 20 - 10 #resize_rate + shift_rate
                face_margin_y1 = 20 + 5 #resize_rate + shift_rate
                face_margin_x2 = -20 - 10 #resize_rate + shift_rate
                face_margin_y2 = -20 + 5 #resize_rate + shift_rate
                face_margin_h = -0.7 #resize_factor
                face_margin_w = -0.7 #resize_factor
            #Adjusting the framePose for profile right
            elif(my_cascade.face_type == 5):
                face_margin_x1 = 20 + 10
                face_margin_y1 = 20 + 5
                face_margin_x2 = -20 + 10
                face_margin_y2 = -20 + 5
                face_margin_h = -0.7
                face_margin_w = -0.7
            #No adjustments
            else:
                face_margin_x1 = 0
                face_margin_y1 = 0
                face_margin_x2 = 0
                face_margin_y2 = 0
                face_margin_h = 0
                face_margin_w = 0

            #Updating the face position
            face_x1 = my_cascade.face_x + roi_x1 + face_margin_x1
            face_y1 = my_cascade.face_y + roi_y1 + face_margin_y1
            face_x2 = my_cascade.face_x + my_cascade.face_w + roi_x1 + face_margin_x2
            face_y2 = my_cascade.face_y + my_cascade.face_h + roi_y1 + face_margin_y2
            face_w = my_cascade.face_w + int(my_cascade.face_w * face_margin_w)
            face_h = my_cascade.face_h + int(my_cascade.face_h * face_margin_h)

            #Updating the ROI position       
            roi_x1 = face_x1 - roi_resize_w
            if (roi_x1 < 0): roi_x1 = 0
            roi_y1 = face_y1 - roi_resize_h
            if(roi_y1 < 0): roi_y1 = 0
            roi_w = face_w + roi_resize_w + roi_resize_w
            if(roi_w > cam_w): roi_w = cam_w
            roi_h = face_h + roi_resize_h + roi_resize_h
            if(roi_h > cam_h): roi_h = cam_h    
            roi_x2 = face_x2 + roi_resize_w
            if (roi_x2 > cam_w): roi_x2 = cam_w
            roi_y2 = face_y2 + roi_resize_h
            if(roi_y2 > cam_h): roi_y2 = cam_h

            #Debugging printing utilities
            if(DEBUG == True):
                print("FACE: ", face_x1, face_y1, face_x2, face_y2, face_w, face_h)
                print("ROI: ", roi_x1, roi_y1, roi_x2, roi_y2, roi_w, roi_h)
                #Drawing a green rectangle
                # (and text) around the face.
                text_x1 = face_x1
                text_y1 = face_y1 - 3
                if(text_y1 < 0): text_y1 = 0
                cv2.putText(framePose, "FACE", (text_x1,text_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1);
                cv2.rectangle(framePose, 
                             (face_x1, face_y1), 
                             (face_x2, face_y2), 
                             (0, 255, 0),
                              2)

            #In case of a frontal/rotated face it
            # is called the landamark detector
            if(my_cascade.face_type > 0):
                landmarks_2D = my_detector.returnLandmarks(framePose, face_x1, face_y1, face_x2, face_y2, points_to_return=TRACKED_POINTS)

                if(DEBUG == True):
                    #cv2.drawKeypoints(framePose, landmarks_2D)

                    for point in landmarks_2D:
                        cv2.circle(framePose,( point[0], point[1] ), 2, (0,0,255), -1)


                #Applying the PnP solver to find the 3D pose
                # of the head from the 2D position of the
                # landmarks.
                #retval - bool
                #rvec - Output rotation vector that, together with tvec, brings 
                # points from the model coordinate system to the camera coordinate system.
                #tvec - Output translation vector.
                retval, rvec, tvec = cv2.solvePnP(landmarks_3D, 
                                                  landmarks_2D, 
                                                  camera_matrix, camera_distortion)



                #To get roll, yaw, pitch
                #Using cv2.Rodrigues(src[, dst[, jacobian]]) → dst, jacobian
                #src – Input rotation vector (3x1 or 1x3) or rotation matrix (3x3).
                #dst – Output rotation matrix (3x3) or rotation vector (3x1 or 1x3), respectively.
                #jacobian – Optional output Jacobian matrix, 3x9 or 9x3, which is a matrix of partial derivatives of the output array components with respect to the input array components.

                #Get as input the rotational vector
                #Return a rotational matrix
                rmat, _ = cv2.Rodrigues(rvec)


                #print(rmat)

                # proMatrix =   np.array([[ rmat[0,0], rmat[0,1], rmat[0,2], 0],
                #                            [ rmat[1,0], rmat[1,1], rmat[1,2], 0],
                #                            [ rmat[2,0], rmat[2,1], rmat[2,2], 0] ])

                #print(proMatrix)

                #euler_angles contain (pitch, yaw, roll)
                
                euler_angles = rotationMatrixToEulerAngles(rmat)


                print(euler_angles)

                yaw = math.degrees(euler_angles[1]);
                pitch = math.degrees(euler_angles[0]);
                roll = math.degrees(euler_angles[2]);
                # yaw = math.cos(euler_angles[1]*180/math.pi)
                # pitch = math.cos(euler_angles[0]*180/math.pi)
                # roll = math.cos(euler_angles[2]*180/math.pi)

                # yaw = euler_angles[6][1];
                # pitch = euler_angles[6][0];
                # roll = euler_angles[6][2];

                print('yaw = '+repr(yaw)+'\n')
                print('pitch = '+repr(pitch)+'\n')
                print('roll = '+repr(roll)+'\n')

                cv2.putText(framePose, "YAW: {:.2f}".format(yaw), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(framePose, "Pitch: {:.2f}".format(pitch), (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # head_pose = [ rmat[0,0], rmat[0,1], rmat[0,2], tvec[0],
                #               rmat[1,0], rmat[1,1], rmat[1,2], tvec[1],
                #               rmat[2,0], rmat[2,1], rmat[2,2], tvec[2],
                #                     0.0,      0.0,        0.0,    1.0 ]
                # #print(head_pose) #TODO remove this line
                 
                
                #print (rotationMatrixToEulerAngles(rmat))
                
                # Calculates rotation matrix to euler angles
                # The result is the same as MATLAB except the order
                # of the euler angles ( x and z are swapped ).

                #assert(isRotationMatrix(R))
 
                #To prevent the Gimbal Lock it is possible to use
                #a threshold of 1e-6 for discrimination
                # sy = math.sqrt(rmat[0,0] * rmat[0,0] +  rmat[1,0] * rmat[1,0])    
                # singular = sy < 1e-6

                # if  not singular :
                #     x = math.atan2(rmat[2,1] , rmat[2,2])
                #     y = math.atan2(-rmat[2,0], sy)
                #     z = math.atan2(rmat[1,0], rmat[0,0])
                # else :
                #     x = math.atan2(-rmat[1,2], rmat[1,1])
                #     y = math.atan2(-rmat[2,0], sy)
                #     z = 0

                # print(np.array([x, y, z]))



                

                #Now we project the 3D points into the image plane
                #Creating a 3-axis to be used as reference in the image.
                axis = np.float32([[50,0,0], 
                                      [0,50,0], 
                                      [0,0,50]])
                imgpts, jac = cv2.projectPoints(axis, rvec, tvec, camera_matrix, camera_distortion)

                #Drawing the three axis on the image framePose.
                #The opencv colors are defined as BGR colors such as: 
                # (a, b, c) >> Blue = a, Green = b and Red = c
                #Our axis/color convention is X=R, Y=G, Z=B
                sellion_xy = (landmarks_2D[7][0], landmarks_2D[7][1])
                cv2.line(framePose, sellion_xy, tuple(imgpts[1].ravel()), (0,255,0), 3) #GREEN
                cv2.line(framePose, sellion_xy, tuple(imgpts[2].ravel()), (255,0,0), 3) #BLUE
                cv2.line(framePose, sellion_xy, tuple(imgpts[0].ravel()), (0,0,255), 3) #RED

        #Drawing a yellow rectangle
        # (and text) around the ROI.
        if(DEBUG == True):
            text_x1 = roi_x1
            text_y1 = roi_y1 - 3
            if(text_y1 < 0): text_y1 = 0
            cv2.putText(framePose, "ROI", (text_x1,text_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1);
            cv2.rectangle(framePose, 
                         (roi_x1, roi_y1), 
                         (roi_x2, roi_y2), 
                         (0, 255, 255),
                         2)

        #Showing the framePose and waiting
        # for the exit command
        cv2.imshow('Video', framePose)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
   
    #Release the camera
    video_capture.release()
    print("Bye...")

######## Head Pose Part Code #########


frameNumber = 0
earArray = []
frameArray = []

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

#animate the plotted data
def animate(i):
    graph_data = open('/home/ashfak/Desktop/DriversAssistanceSystem/deepgaze/FaceProject/test.txt', 'r').read()
    lines = graph_data.split('\n')
    xs = []
    ys = []
    for line in lines:
        if len(line) > 1:
            x, y = line.split(',')
            xs.append(x)
            ys.append(y)
    ax1.clear()
    ax1.plot(xs, ys)
    plt.axhline(y=EYE_AR_THRESH, c="red")
    ax1.scatter(xs, ys)

def animator():
	print("[INFO] loading graph...")
	ani = animation.FuncAnimation(fig, animate, interval=50)
	plt.show()



# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--shape-predictor", required=True,
# 	help="path to facial landmark predictor")
# ap.add_argument("-a", "--alarm", type=str, default="",
# 	help="path alarm .WAV file")
# ap.add_argument("-w", "--webcam", type=int, default=0,
# 	help="index of webcam on system")
# args = vars(ap.parse_args())
 
# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold for to set off the
# alarm
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 48

# initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off
COUNTER = 0
ALARM_ON = False



# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(args["shape_predictor"])
predictor = dlib.shape_predictor('/home/ashfak/Desktop/DriversAssistanceSystem/deepgaze/FaceProject/shape_predictor_68_face_landmarks.dat')

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
(m,n) =(1,68)

# start the video stream thread
print("[INFO] starting video stream thread...")
# vs = VideoStream(src=args["webcam"]).start()
vs = VideoStream(src=0).start()
time.sleep(1.0)


f = open('/home/ashfak/Desktop/DriversAssistanceSystem/deepgaze/FaceProject/test.txt', 'r+')
f.truncate()

# loop over frames from the video stream
while True:
	
	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a np
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		mouth = shape[mStart:mEnd]
		nose = shape[nStart:nEnd]
		allpoints = shape[m:n]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0

		# frameNumber = frameNumber + 1
		# plotAr = np.empty(1)
		# plotArray = np.append(plotAr, [[frameNumber, ear]])

		frameNumber = frameNumber + 1
		frameArray.append(frameNumber)
		earArray.append(ear)
		len1 = len(frameArray)
		len2 = len(earArray)

		#print(earArray, frameArray)
		#ani = animation.FuncAnimation(fig, animate, fargs=(frameArray, earArray),  interval=50, blit=True)
		#plt.show()
		#
		file = open("/home/ashfak/Desktop/DriversAssistanceSystem/deepgaze/FaceProject/test.txt","a")
		file.write(str(frameNumber) + "," + str(ear) +"\n")

		if(frameNumber > 60):
			with open('/home/ashfak/Desktop/DriversAssistanceSystem/deepgaze/FaceProject/test.txt', 'r') as fin:
			    data = fin.read().splitlines(True)
			with open('/home/ashfak/Desktop/DriversAssistanceSystem/deepgaze/FaceProject/test.txt', 'w') as fout:
			    fout.writelines(data[1:])

		


		# print(plotArray)
		#ani = animation.FuncAnimation(fig, animate, fargs=(frameArray, earArray),  interval=50)
		#plt.show()

		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		mouthHull = cv2.convexHull(mouth)
		noseHull = cv2.convexHull(nose)
		allHull = cv2.convexHull(allpoints)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [noseHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [allHull], -1, (0, 255, 0), 1)

		# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
		if ear < EYE_AR_THRESH:
			COUNTER += 1

			# if the eyes were closed for a sufficient number of
			# then sound the alarm
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				# if the alarm is not on, turn it on
				if not ALARM_ON:
					ALARM_ON = True

					# check to see if an alarm file was supplied,
					# and if so, start a thread to have the alarm
					# sound played in the background
					# if args["alarm"] != "":
					t = Thread(target=sound_alarm,
							args=('/home/ashfak/Desktop/DriversAssistanceSystem/deepgaze/FaceProject/alert.wav',))
					t.deamon = True
					t.start()

				# draw an alarm on the frame
				cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		# otherwise, the eye aspect ratio is not below the blink
		# threshold, so reset the counter and alarm
		else:
			COUNTER = 0
			ALARM_ON = False

		# draw the computed eye aspect ratio on the frame to help
		# with debugging and setting the correct eye aspect ratio
		# thresholds and frame counters
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
	# show the frame
	cv2.imshow("Frame", frame)
	t1 = Thread(target=headPose)
	t1.deamon = True
	t1.start()



	
	
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break


# do a bit of cleanup

cv2.destroyAllWindows()
vs.stop()