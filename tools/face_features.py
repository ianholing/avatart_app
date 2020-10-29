from collections import OrderedDict
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

FACIAL_LANDMARKS_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 35)),
	("jaw", (0, 17))
])

def extract_face_features(image):
	scale_size = 256
	correction = 1.0*image.shape[1]/scale_size
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor("tools/shape_predictor_68_face_landmarks.dat")
	
	image = imutils.resize(image, width=scale_size)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	rects = detector(gray, 1)

	faces = []
	for (i, rect) in enumerate(rects):
		features = {}
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
	
		for (name, (i, j)) in FACIAL_LANDMARKS_IDXS.items():
			features[name] = shape[i:j] * correction

		faces.append(features)
	
	return faces

def angle_between(l, r):
	w = r[0] - l[0]
	h = r[1] - l[1]
	angle = np.arctan2(h, w) 
	return (np.rad2deg(angle) + 180) % 360

def point_after_rotation(p, angle, width, height):
	x, y = p
	x, y = x - width/2, y - height/2
	y = y*np.cos(np.deg2rad(angle)) - x*np.sin(np.deg2rad(angle))
	x = y*np.sin(np.deg2rad(angle)) + x*np.cos(np.deg2rad(angle))
	x, y = x + width/2, y + height/2
	return x, y