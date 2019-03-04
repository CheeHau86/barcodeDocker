from django.shortcuts import render
from django.http import Http404
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from django.core import serializers
from django.conf import settings

import sys
import io
import math
import os,glob,cv2
import argparse
import numpy as np
import random
import zxing
import imutils
import time
import json
import base64

#
# @brief function to offset the roi images to feed to zxing barcode reader
# @param full image, roi, offset boolean, median value of all bounding boxes's height, contours' poly to calulcate rotation, boolean to append barcode's stop header, stop header's image from other barcode in the same image
# @return decoded barcode's information

# @django json post and response CheckBarcode function

def roi_offset(image, rect, offset, median_value, contours_poly, append, stop_header_image):
	information = ""
	zx = zxing.BarCodeReader() #Initialize Barcode Reader

	x,y,w,h = rect
	if(offset): # offset the position of bounding box of the barcode
		y_offset = rect[3]-int(4*median_value/5)
		y = y + y_offset
		h = h - int(y_offset)

	rect = (x,y,w,h)
	roi_image = image[rect[1]:(rect[1]+rect[3]), rect[0]:(rect[0]+rect[2])]
	roi_image = cv2.resize(roi_image, (500, 150),0,0, cv2.INTER_LINEAR) # resize image to enable better detection
	roi_image = rotate_image(contours_poly,roi_image)

	if(append): # append the stop header
		roi_image_1 = sharpened_gray[rect[1]:(rect[1]+rect[3]), rect[0]:(rect[0]+rect[2])]
		roi_image_1 = cv2.resize(roi_image_1, (500, 150),0,0, cv2.INTER_LINEAR)
		roi_image_1 = rotate_image(contours_poly,roi_image_1)
		#bounding_rect = find_contour(roi_image_1)

		#bounding_box_width_end = rect[0]+rect[2]
		#bounding_rect_width_end = bounding_rect[0] + bounding_rect[2]

		stop_header_image = cv2.resize(stop_header_image, (20, 150),0,0, cv2.INTER_LINEAR)
		roi_image[:,roi_image.shape[1]-stop_header_image.shape[1]:] = stop_header_image # append the end of the image with the stop header

	# Decode Barcode
	cv2.imwrite("MyApp/detection_images/roi_image_1.jpg",roi_image) #./
	test_image1 = "MyApp/detection_images/roi_image_1.jpg" #./
	barcode = zx.decode(test_image1, True)
	if(barcode is not None):
		information = barcode.raw
		print("Function_1: " + information)

	return(information)

#
# @brief function to offset lesser of the roi images to feed to zxing barcode reader
# @param full image, roi, offset boolean, median value of all bounding boxes's height, contours' poly to calulcate rotation, boolean to append barcode's stop header, stop header's image from other barcode in the same image
# @return decoded barcode's information
def roi_offset_less(image, rect, offset,median_value, contours_poly, append, stop_header_image):
	information = ""
	zx = zxing.BarCodeReader()
	x,y,w,h = rect
	if(offset):
		y_offset = rect[3]-int(1*median_value/4)
		y = y + y_offset
		h = median_value

	rect = (x,y,w,h)
	roi_image = image[rect[1]:(rect[1]+rect[3]), rect[0]:(rect[0]+rect[2])]
	roi_image = cv2.resize(roi_image, (500, 150),0,0, cv2.INTER_LINEAR)
	roi_image = rotate_image(contours_poly,roi_image)

	if(append):

		roi_image_2 = sharpened_gray[rect[1]:(rect[1]+rect[3]), rect[0]:(rect[0]+rect[2])]
		roi_image_2 = cv2.resize(roi_image_2, (500, 150),0,0, cv2.INTER_LINEAR)
		roi_image_2 = rotate_image(contours_poly,roi_image_2)
		#bounding_rect = find_contour(roi_image_2)

		#bounding_box_width_end = rect[0]+rect[2]
		#bounding_rect_width_end = bounding_rect[0] + bounding_rect[2]

		stop_header_image = cv2.resize(stop_header_image, (30, 150),0,0, cv2.INTER_LINEAR)
		roi_image[:,roi_image.shape[1]-stop_header_image.shape[1]:] = stop_header_image # append the end of the image with the stop header

	# Decode Barcode
	cv2.imwrite("MyApp/detection_images/roi_image_2.jpg",roi_image)	#./
	test_image2 = "MyApp/detection_images/roi_image_2.jpg" #./
	barcode = zx.decode(test_image2, True)
	if(barcode is not None):
		information = barcode.raw
		print("Function_2: " + information)

	return(information)

#
# @brief function to rotate the image based on it's contours
# @param contours and image
# @return the rotated image
def rotate_image(contours_poly,roi_image):
	angle = cv2.minAreaRect(contours_poly)[-1] #obtain the angle of the rectangle
	if angle < -45:
		angle = -(90 + angle)
	else:
		angle = -angle

	if(abs(angle) >= 1): #if the angle is larger than 1. rotate the image
		rotated = imutils.rotate_bound(roi_image, angle)
		# Resize the image back to it's original size
		rotated = cv2.resize(rotated, (roi_image.shape[1], roi_image.shape[0]),0,0, cv2.INTER_LINEAR)
		cv2.imwrite("MyApp/detection_images/rotated.jpg",rotated) #./
	else:
		rotated = roi_image

	return rotated

#
# @brief function to group bounding boxes together. Primarly use for group the stop headers
# @param list of bounding boxes
# @return one combine bounding box
def sum_bounding_box(bounding_box):
	min_y = float('Inf')
	max_h = 0
	for i in range(0, len(bounding_box)):
		x,y,w,h =  bounding_box[i]
		if(y < min_y):
			min_y = y
		if((y+h) > max_h):
			max_h = y+h

	start_x = bounding_box[0][0]-5  #12
	end_x = bounding_box[-1][0] + bounding_box[-1][2]
	width = end_x - start_x + 7
	height = max_h + 20
	rect = (start_x, min_y, width, height)

	return(rect)

#
# @brief function to find and segment each "bar" in the barcodes into bounding boxes
# @param the roi image of the barcode
# @return the last sorted bounding box
def find_contour(roi_image_barcode):
	roi_barcode_thres = cv2.adaptiveThreshold(~roi_image_barcode,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,-15)
	roi_barcode_thres = cv2.bitwise_not(roi_barcode_thres)
	cv2.imwrite("MyApp/detection_images/roi_barcode_threshold_2.jpg",roi_barcode_thres) #./

	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
	closed = cv2.morphologyEx(roi_barcode_thres, cv2.MORPH_CLOSE, kernel)
	cv2.imwrite("MyApp/detection_images/barcode_filtered_2.jpg", closed) #./

	im2, contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	draw_image = roi_image_barcode.copy()
	bounding_box_contour = []
	for m in range (0, len(contours)):
		color = []
		b = random.randint(0, 255)
		g = random.randint(0, 255)
		r = random.randint(0, 255)
		color = [b,g,r]
		if(cv2.contourArea(contours[m]) > 20): #needs to be tuned based on image resolution
			cv2.drawContours(draw_image, contours, m, color, 1)
			bounding_box_contour.append((cv2.boundingRect(contours[m])))

	sorted_bounding_box = sorted(bounding_box_contour, key=lambda b:b[0])
	cv2.imwrite("MyApp/detection_images/barcode_contour_image_2.jpg", draw_image)	#./
	return(sorted_bounding_box[-1])

#
# @brief function to write the decoded information onto the image
# @param image, detected bounding boxes, decoded information
# @return
def write_to_image(image,detected_bounding_box,detected_information):
	# Write the decoded barcode the Image
	for i in range(0, len(detected_bounding_box)):
		color = []
		b = random.randint(0, 255)
		g = random.randint(0, 255)
		r = random.randint(0, 255)
		color = [b,g,r]

		rect = detected_bounding_box[i]
		cv2.rectangle(image_bgr_clean_2, (rect[0],rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), color, 2, 8, 0)

		cv2.putText(image, detected_information[i],(rect[0]+int(rect[2]/8),rect[1]+int(rect[3]/2)), cv2.FONT_HERSHEY_SIMPLEX,
					1,(0,0,255),3,cv2.LINE_AA)

	cv2.imwrite("MyApp/detection_images/final_segmented_barcode.jpg", image) #./

#
# @brief function to write the decoded information into a json file
# @param output filename, decoded information
# @return
def write_to_json_file(output_filename,detected_information):
	# Write to Json file
	detection_flag = False
	if(any("PNR" in info for info in detected_information) and any("SER" in info for info in detected_information)):
		detection_flag = True
	PNR =()
	SER =()
	PNR = [info for info in detected_information if "PNR" in info]
	SER = [info for info in detected_information if "SER" in info]

	#convert list to string for json response
	global SERN, PNRN
	SERN = str(SER)
	PNRN = str(PNR)
	#-convert list to string for json response

	data = {
		"PNR and SER Detection":str(detection_flag),
		"PNR":str(PNR)[2:-2],
		"SER":str(SER)[2:-2]
	}

	with open(output_filename, 'w') as outfile:
		json.dump(data, outfile)

#
# @brief function to read config file (JSON)
# @param config file's directory
# @return the input image and output filename
def read_config(config_file):
	with open(config_file) as json_file:
		data = json.load(json_file)
		input_image = data["Input File"]
		output_filename = data["Output File"]

	return input_image, output_filename


# @brief function to read JSON msg
# @param JSON msg file directory
# @return the image in base64, SER and PNR
def read_JSON_msg(JSON_file):
	with open(JSON_file) as json_file:
		data = json.load(json_file)
		image_base64 = data["image_base64"]
		#SER = data["SER"]
		#PNR = data["PNR"]

	return image_base64 #, SER, PNR

@api_view(["POST"])
def CheckBarcode(barcodedata):

	image_base64=json.loads(barcodedata.body)

	# Read Config from JSON File
	#test_image, output_filename = read_config("D:/001 TanCH/001 Files/001 Project/106 Radome/Airbus S'pore/codes/sae_barcode_detector_v1/DjangoBarcode/SampleProject/MyApp/config.txt") #try read from json image base64 string
	#output filename
	output_filename = "../barcodeDetector/data.txt"
	input_filename = "../barcodeDetector/json_input.txt"

	#print("Input File: ", str(sys.argv[1])) #"/home/haofucv-2/AIRBUS/EAST/Radome Labels/img_187.JPG"
	#testimage = sys.argv[1]

	#image_base64, SER, PNR = read_JSON_msg("D:/001 TanCH/001 Files/001 Project/106 Radome/Airbus S'pore/codes/sae_barcode_detector_v1/DjangoBarcode/SampleProject/MyApp/JSON_msg.txt")
	#image_base64 = read_JSON_msg(input_filename)
	#image_base64 = read_JSON_msg("D:/001 TanCH/001 Files/001 Project/106 Radome/Airbus S'pore/codes/sae_barcode_detector_v1/DjangoBarcode/SampleProject/MyApp/JSON_msg.txt")

	#Encode image to base64 (simulate JSON msg)
	#retval, buffer = cv2.imencode('.jpg', image_bgr)
	#jpg_as_text = base64.b64encode(buffer)
	#print(jpg_as_text)

	#Decode base64 to image
	imgdata = base64.b64decode(image_base64)
	filename = '../barcodeDetector/converted_JSON.jpg'  # I assume you have a way of picking unique filenames
	with open(filename, 'wb') as f:
		f.write(imgdata)

	#converted_JSON = cv2.imread(filename, cv2.IMREAD_COLOR)
	# Read the Image
	image_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
	if image_gray is None:
		print("ERROR: Unable to Read Image!!")
		sys.exit()

	image_gray_clean = image_gray.copy()

	image_bgr = cv2.imread(filename, cv2.IMREAD_COLOR)

	image_bgr_clean = image_bgr.copy()
	global image_bgr_clean_2
	image_bgr_clean_2 = image_bgr.copy()

	# Sharpening of the Image
	global kernel1
	kernel1 = np.zeros( (9,9), np.float32)
	kernel1[4,4] = 2.0   #Identity, times two!
	boxFilter = np.ones( (9,9), np.float32) / 81.0
	kernel1 = kernel1 - boxFilter
	sharpened_bgr_1 = cv2.filter2D(image_bgr, -1, kernel1)
	cv2.imwrite("../barcodeDetector/MyApp/detection_images/sharpened.jpg", sharpened_bgr_1) #./

	# Sharpening of the Image
	kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
	sharpened_bgr_2 = cv2.filter2D(image_bgr, -1, kernel)
	cv2.imwrite("../barcodeDetector/MyApp/detection_images/sharpened_bgr.jpg", sharpened_bgr_2) #./

	# Sharpening of the Grayscale Image
	global sharpened_gray
	sharpened_gray = cv2.filter2D(image_gray, -1, kernel1)
	cv2.imwrite("../barcodeDetector/MyApp/detection_images/sharpened_gray.jpg", sharpened_gray) #./

	scale = 800.0 / image_gray.shape[1] #Resize image to 800 width while keeping the aspect ratio
	image_gray_resized = cv2.resize(image_gray, (int(image_gray.shape[1] * scale), int(image_gray.shape[0] * scale)))
	#cv2.imwrite("resized_gray.jpg",image_gray_resized)

	barcode_detection_start_time = time.time()

	# Perform Morphological Transformation
	kernel = np.ones((1, 3), np.uint8)
	image_gray_resized = cv2.morphologyEx(image_gray_resized, cv2.MORPH_BLACKHAT, kernel, anchor=(1, 0))

	# Threshold the Image
	image_gray_thres = cv2.adaptiveThreshold(image_gray_resized,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,-4)
	cv2.imwrite("../barcodeDetector/MyApp/detection_images/threshold.jpg",image_gray_thres) #./

	# Perform Morphological Transformation again (Dilate and Erode) to connect the compenents
	kernel = np.ones((1, 3), np.uint8)
	image_gray_thres = cv2.morphologyEx(image_gray_thres, cv2.MORPH_DILATE, kernel, anchor=(2, 0), iterations=1)
	image_gray_thres = cv2.morphologyEx(image_gray_thres, cv2.MORPH_DILATE, kernel, anchor=(1, 0), iterations=5)
	image_gray_thres = cv2.morphologyEx(image_gray_thres, cv2.MORPH_CLOSE, kernel, anchor=(1, 0), iterations=3)
	cv2.imwrite("../barcodeDetector/MyApp/detection_images/connected_components.jpg",image_gray_thres) #./

	# Filtering the small elements
	kernel = np.ones((10, 15), np.uint8)
	image_gray_thres = cv2.morphologyEx(image_gray_thres, cv2.MORPH_OPEN, kernel, iterations=1)
	cv2.imwrite("../barcodeDetector/MyApp/detection_images/refined_connected_components.jpg",image_gray_thres) #./

	# Find the contours for the remaining connected components
	im2,contours, hierarchy = cv2.findContours(image_gray_thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	unscale = 1.0 / scale
	bounding_box = []
	contours_poly = []
	height = []
	box_list = []
	if contours != None:
		for j in range(1, len(contours)):

			if(cv2.contourArea(contours[j]) <= 500): #Proceed only if the area of the contours is >=500
				continue
			contours_poly.append(cv2.approxPolyDP(contours[j],3,True))

		for k in range(0, len(contours_poly)):
			color = []
			b = random.randint(0, 255)
			g = random.randint(0, 255)
			r = random.randint(0, 255)
			color = [b,g,r]

			# Draw the minimum detected bounding boxes of in red
			x,y,w,h = cv2.boundingRect(contours_poly[k])
			rect1 = cv2.minAreaRect(contours_poly[k])
			box = cv2.boxPoints(rect1)
			box = np.int0(box*unscale)

			box_list.append(box)
			image_bgr = cv2.drawContours(image_bgr,[box],0,(0,0,255),2)

			# Draw the offset-ed bounding boxes of in different color (this is done to provided more boundary to the bounding boxes
			x_roi_offset = 2
			y_roi_offset = 2
			if((x-x_roi_offset)<0):
				x_roi_offset = x_roi_offset - abs(x-x_roi_offset)

			if((y-y_roi_offset)<0):
				y_roi_offset = y_roi_offset - abs(y-y_roi_offset)

			rect = (int((x-x_roi_offset)*unscale), int((y-y_roi_offset)*unscale), int((w+5)*unscale), int((h+10)*unscale))
			cv2.rectangle(image_bgr, (rect[0],rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), color, 1, 8, 0)

			bounding_box.append(rect) #store these offset-ed bounding boxes
			height.append(rect[3]) #store the height of these bounding boxes

	cv2.imwrite("../barcodeDetector/MyApp/detection_images/segmented_barcode.jpg",image_bgr) #./
	duration = time.time() - barcode_detection_start_time
	print('-----[Segmented Barcode Detection Timing] {:.4f} seconds -----'.format(duration))

	median_value = int(np.median(height))

	detection_list = ["MFR", "SER", "DMF", "PNR"]

	image_list = []
	image_list.append(sharpened_bgr_1)
	#image_list.append(image_bgr_clean)
	#image_list.append(image_gray_clean)
	#image_list.append(sharpened_bgr_2)

	offset_list = [True, False]

	detected_bounding_box = []
	detected_contour_poly = []
	detected_information = []
	undetected_bounding_box = []
	undetected_contour_poly = []
	print('[No of Barcodes Detected] {:d}'.format(len(bounding_box)))
	barcode_decoding_start_time = time.time()
	for i in range(0, len(bounding_box)):
		x,y,w,h = bounding_box[i]
		found = False

		for j in range(0, len(image_list)): # iterate through the different type of images

			for k in range(0, len(offset_list)): # iterate through the different offset condition
				rect = (x,y,w,h)
				# Run decoding twice (different offset of the roi)
				information_1 = roi_offset(image_list[j], rect, offset_list[k], median_value, contours_poly[i], False, None)

				information_2 = roi_offset_less(image_list[j], rect, offset_list[k], median_value, contours_poly[i], False, None)

				# Check if the detected information is part of the detection list
				if(information_1[:3] in detection_list):
					found = True
					information = information_1
					break
				elif(information_2[:3] in detection_list):
					found = True
					information = information_2
					break

			if(found):
				break

		# If the detected information is part of the detected list, store the bounding box, contours and decoded information of the barcode
		if(found):
			detected_bounding_box.append(bounding_box[i])
			detected_contour_poly.append(contours_poly[i])
			detected_information.append(information)
			# Remove from the detect list
			detection_list.remove(information[:3])
			continue
		else:
			undetected_bounding_box.append(bounding_box[i])
			undetected_contour_poly.append(contours_poly[i])

	# For the detected barcodes, measure the min width
	min_width = float('Inf')
	for i in range(0, len(detected_bounding_box)):
		if(detected_bounding_box[i][2] < min_width):
			min_width = detected_bounding_box[i][2]


	# For the undetected bounding boxes, check if the width is within certain threshold of the mininum detected
	# bounding box's width - this is done to remove false positives
	new_undeteced_bounding_box = []
	new_undetected_contour_poly = []

	for j in range(0, len(undetected_bounding_box)):
		width = undetected_bounding_box[j][2]
		if((width + float(min_width/2))> min_width):
			new_undeteced_bounding_box.append(undetected_bounding_box[j])
			new_undetected_contour_poly.append(undetected_contour_poly[j])

	# Extract stop headers from succesful detected barcodes
	stop_header_bounding_box = []
	for l in range(0,len(detected_bounding_box)):

		roi_barcode = image_gray_clean[detected_bounding_box[l][1]:(detected_bounding_box[l][1]+detected_bounding_box[l][3]),
									   detected_bounding_box[l][0]:(detected_bounding_box[l][0]+detected_bounding_box[l][2])]

		roi_barcode_bgr = image_bgr_clean[detected_bounding_box[l][1]:(detected_bounding_box[l][1]+detected_bounding_box[l][3]),
									   detected_bounding_box[l][0]:(detected_bounding_box[l][0]+detected_bounding_box[l][2])]

		# Rotate Image
		rotated_roi_image = rotate_image(detected_contour_poly[l],roi_barcode)
		rotated_roi_image_bgr = rotate_image(detected_contour_poly[l],roi_barcode_bgr)
		rotated_roi_image_bgr_copy = rotated_roi_image_bgr.copy()

		# Thresholding
		roi_barcode_thres = cv2.adaptiveThreshold(~rotated_roi_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,-4)
		roi_barcode_thres_copy = roi_barcode_thres.copy()
		roi_barcode_thres = cv2.bitwise_not(roi_barcode_thres)
		cv2.imwrite("../barcodeDetector/MyApp/detection_images/roi_barcode_threshold.jpg",roi_barcode_thres) #./

		# Morphological Transformation (Dilate and Erode)
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
		closed = cv2.morphologyEx(roi_barcode_thres, cv2.MORPH_CLOSE, kernel)
		cv2.imwrite("../barcodeDetector/MyApp/detection_images/barcode_filtered.jpg", closed) #./

		# Find the contours of the Barcode and its bounding boxes
		im2, contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		bounding_box_contour = []
		refined_contour = []
		for m in range (0, len(contours)):
			color = []
			b = random.randint(0, 255)
			g = random.randint(0, 255)
			r = random.randint(0, 255)
			color = [b,g,r]
			if(cv2.contourArea(contours[m]) > 20):
				bounding_box_contour.append((cv2.boundingRect(contours[m])))
				refined_contour.append(contours[m])

		# Sort the bounding boxes from left to right order
		sorted_bounding_box = sorted(bounding_box_contour, key=lambda b:b[0])
		last_3_bounding_box = []
		# Look at the last 3 bounding boxes
		for n in range (len(sorted_bounding_box)-3,len(sorted_bounding_box)):
			color = []
			b = random.randint(0, 255)
			g = random.randint(0, 255)
			r = random.randint(0, 255)
			color = [b,g,r]
			rect = sorted_bounding_box[n]

			last_3_bounding_box.append(rect)
		# Sum the last 3 bounding boxes into one big bounding box
		rect = sum_bounding_box(last_3_bounding_box)
		stop_header_bounding_box.append(rect)
		cv2.rectangle(rotated_roi_image_bgr_copy, (rect[0],rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), [0,0,255], 1, 8, 0)
		cv2.imwrite("../barcodeDetector/MyApp/detection_images/barcode_bb_image.jpg", rotated_roi_image_bgr_copy) #./

	# Second Round of Detection
	if (len(detection_list)>0): #detection list is not empty yet
		print("---- Starting second round of detection -----")
		for i in range(0, len(new_undeteced_bounding_box)):
			found = False
			x,y,w,h = new_undeteced_bounding_box[i]
			for o in range(0, len(stop_header_bounding_box)): # Loop through the number of stop headers extracted

				for p in range(0, len(image_list)): # Loop through the type of images (brg, gray, sharpened and e.g.) however we only loop through the sharpened image

					for q in range(0, len(offset_list)): # Loop through whether to offset or not
						image_roi = image_list[p][detected_bounding_box[o][1]:(detected_bounding_box[o][1]+detected_bounding_box[o][3]),
													detected_bounding_box[o][0]:(detected_bounding_box[o][0]+detected_bounding_box[o][2])]

						rotated_roi_image = rotate_image(detected_contour_poly[o],image_roi)
						roi_stop_header_image = rotated_roi_image[stop_header_bounding_box[o][1]:(stop_header_bounding_box[o][1]+stop_header_bounding_box[o][3]),
															stop_header_bounding_box[o][0]:(stop_header_bounding_box[o][0]+stop_header_bounding_box[o][2])]

						# Run decoding twice (different offset of the roi)
						rect = (x,y,w,h)
						information_1 = roi_offset(image_list[p], rect, offset_list[q], median_value, new_undetected_contour_poly[i], True, roi_stop_header_image)

						information_2 = roi_offset_less(image_list[p], rect, offset_list[q], median_value, new_undetected_contour_poly[i], True, roi_stop_header_image)


						if(information_1[:3] in detection_list):
							found = True
							information = information_1
							break
						elif(information_2[:3] in detection_list):
							found = True
							information = information_2
							break

					if(found):
						break

				if(found):
					break

			if(found):
				detected_bounding_box.append(new_undeteced_bounding_box[i])
				detected_information.append(information)
				# Remove from list
				detection_list.remove(information[:3])
				if(len(detection_list) == 0):
					break

	decoding_duration = time.time() - barcode_decoding_start_time
	print('----- [Segmented Barcode Decoding Timing] {:.4f} seconds -----'.format(decoding_duration))

	# Write to Image
	write_to_image(image_bgr_clean_2, detected_bounding_box, detected_information)
	# Write to JSON File
	write_to_json_file(output_filename,detected_information)

	return JsonResponse("SER: "+SERN+" ; PNR: "+PNRN,safe=False)