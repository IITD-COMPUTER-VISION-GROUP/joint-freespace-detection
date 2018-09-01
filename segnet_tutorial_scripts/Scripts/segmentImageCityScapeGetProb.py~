import numpy as np
#import matplotlib.pyplot as plt
import os.path
import scipy
import argparse
import math
import cv2
import sys
import time


#sys.path.append('/usr/local/lib/python2.7/site-packages')
# Make sure that caffe is on the python path:
# caffe_root = '~/caffe-segnet/'
# sys.path.insert(0, caffe_root + 'python')
import caffe


def getProb(arr , roadFile , nonRoadFile):
	# arr = np.load('arr.npy')

	m1 = np.zeros((arr.shape[2],arr.shape[3]), dtype=np.uint8)
	m2 = np.zeros((arr.shape[2],arr.shape[3]), dtype=np.uint8)

	for i in xrange(0,arr.shape[2]):
		for j in xrange(0,arr.shape[3]):
			x = arr[0,:,i,j]
			minVal = min(x)
			if minVal < 0:
				x = x - minVal + 1
			x /= sum(x)
			m1[i,j] = x[4]*100

			m2[i,j] = 100*max( max(x[0:4]), max(x[5:arr.shape[1]]) )


	cv2.imwrite(roadFile,m1)
	cv2.imwrite(nonRoadFile,m2)


# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--colours', type=str, required=True)
parser.add_argument('--input', type=str, required=True)
args = parser.parse_args()

# print args
inputLocation = args.input
inputSplit = inputLocation.split('/')
inputSplit[-2] = inputSplit[-2]+"_SegnetCityS"

inputFile = inputSplit[-1]
inputFileSplit = inputFile.split('.')
inputFileSplit[-1] = 'png'
inputSplit[-1] = '.'.join(inputFileSplit)

roadFileSplit = list(inputSplit)
roadFileSplit[-2] += "_road" 
roadFile = '/'.join(roadFileSplit)

NonroadFileSplit = list(inputSplit)
NonroadFileSplit[-2] += "_Nonroad"
nonRoadFile = '/'.join(NonroadFileSplit) 


print inputSplit


output_dir_split=inputSplit[0:(len(inputSplit)-1)]
output_dir="/".join(output_dir_split)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


output_dir_split=roadFileSplit[0:(len(roadFileSplit)-1)]
output_dir="/".join(output_dir_split)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


output_dir_split=NonroadFileSplit[0:(len(NonroadFileSplit)-1)]
output_dir="/".join(output_dir_split)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)



output = "/".join(inputSplit)


net = caffe.Net(args.model,
                args.weights,
                caffe.TEST)

# net = caffe.Net("../Example_Models/segnet_model_driving_webdemo.prototxt",
#                 "../weights/segnet_weights_driving_webdemo.caffemodel",
#                 caffe.TEST)

# net = caffe.Net("Example_Models/segnet_model_driving_webdemo.prototxt",
#                 "weights/segnet_weights_driving_webdemo.caffemodel",
#                 caffe.TEST)


# caffe.set_mode_gpu()
caffe.set_mode_cpu()

input_shape = net.blobs['data'].data.shape
output_shape = net.blobs['argmax'].data.shape

label_colours = cv2.imread(args.colours).astype(np.uint8)

# cv2.namedWindow("Input")
# cv2.namedWindow("SegNet")

# cap = cv2.VideoCapture(0) # Change this to your webcam ID, or file name for your video file

# if cap.isOpened(): # try to get the first frame
#     rval, frame = cap.read()
# else:
#     rval = False

# while rval:
start = time.time()
# rval, frame = cap.read()
frame = cv2.imread(args.input)
# frame = cv2.imread('/home/shashank/Documents/deepL/SegNet-Tutorial-master/Scripts/segnet_test.png')
# print frame.shape
end = time.time()
print '%30s' % 'Grabbed camera frame in ', str((end - start)*1000), 'ms'

start = time.time()
frame = cv2.resize(frame, (input_shape[3],input_shape[2]))
input_image = frame.transpose((2,0,1))
# input_image = input_image[(2,1,0),:,:] # May be required, if you do not open your data with opencv
input_image = np.asarray([input_image])
end = time.time()
print '%30s' % 'Resized image in ', str((end - start)*1000), 'ms'

start = time.time()
out = net.forward_all(data=input_image)

arr = net.blobs['conv1_1_D'].data
getProb(arr , roadFile , nonRoadFile)
# print "Probs"  
# print probs.shape

# np.save('arr',probs)

end = time.time()
print '%30s' % 'Executed SegNet in ', str((end - start)*1000), 'ms'

print output_shape
start = time.time()
segmentation_ind = np.squeeze(net.blobs['argmax'].data)
segmentation_ind_3ch = np.resize(segmentation_ind,(3,input_shape[2],input_shape[3]))
segmentation_ind_3ch = segmentation_ind_3ch.transpose(1,2,0).astype(np.uint8)
segmentation_rgb = np.zeros(segmentation_ind_3ch.shape, dtype=np.uint8)

cv2.LUT(segmentation_ind_3ch,label_colours,segmentation_rgb)
cv2.imwrite(output,segmentation_ind_3ch)
# cv2.imwrite(output,segmentation_rgb)
segmentation_rgb = segmentation_rgb.astype(float)/255
# cv2.imwrite(output,segmentation_ind)
# cv2.imwrite("output.png",segmentation_ind)

end = time.time()
print '%30s' % 'Processed results in ', str((end - start)*1000), 'ms\n'

# print segmentation_ind_3ch

# cv2.imshow("Input", frame)
# cv2.imshow("SegNet", segmentation_rgb)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# key = cv2.waitKey(1)
# if key == 27: # exit on ESC
#     break
# cap.release()
# cv2.destroyAllWindows()
