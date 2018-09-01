#!/bin/sh

direc=`pwd`
fullpath="/home/shashank/pranjal/potholes/p0/images/"

for i in $(ls $fullpath ); do
	time python Scripts/segmentImageCityScapeGetProb.py --model Example_Models/segnet_model_driving_webdemo_12class.prototxt --weights weights/segnet_weights_driving_webdemo.caffemodel --colours Scripts/camvid12.png --input $fullpath$i
done





