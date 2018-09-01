# joint-freespace-detection
A Joint 3D-2D based Method for Free Space Detection on Roads

This code performs the CRF optimization for 3D-2D based Joint Freespace detection on Roads
This code can either do a joint 3D -2D based optimzation or a pure 2D optimazion based on colorlines By changing the depth weights to 0

REQUIREMENTS: SEGNET, OpenCV 

Inputs required: 
1. Segmentation from Segnet
2. Depth priors either from SLAM or LiDAR data or any other form 


INSTRUCTIONS for Ubuntu

STEP 1: INSTALL SegNet-Tutorial 
Link : https://github.com/alexgkendall/SegNet-Tutorial



Step 2: Merge the scripts and folders given in the folder "segnet_tutorial_scripts" with the folder "Segnet-Tutorial-Master" that is after installation of Step 1



Step 3 (2D priors from SEGNET): The original Segnet code is meant for segmenting Images one by one We have provided a script "segmentAllImages_script.sh" 

Keep all the images in the "images" directory inside your data directory

Change the images directory inside the script by providing the path to the "images" directory.

Run it by the command "bash segmentAllImages_script.sh"



Step 4: Copy the folders (if not already copied): images_SegnetCityS, images_SegnetCityS_Nonroad, images_SegnetCityS_road



Step 5 Change Parameters in src/main.cpp: Default Parmeters already set

K:  change here for 2D weight

B1 change here for weight of 3D component -> [0,1]; 0 means zero weight (2D only mode)

S1: [0,1]; smoothness component (from color lines)



Step 5: Install the joint segmentation code

REQUIREMENTS: OpenCV

STEPS:

mkdir build

cd build

cmake ..

make



Step 6: Providing the depth output (Choice of User SLAM/Lidar/SfM):

Create a folder in the data directory with the name "img_depth"

Based on the input from Lidar/Sonar/SLAM the depth has to be converted into priors and provided to the code

The depth output has to be provided in the form of an image which is the projection of the road and other priors (from depth) on the current camera using the following convention:

For each pixel falling on the road plane/prior, it must be made 0 else for obstacles it has to be made  non zero (e.g. 255). "Samples provided" 

Each of the images need to be placed in the "img_depth" folder with the same name as the original image (e.g. img_24.jpg.png as in the example)



Step 7: 
Running for a dataset:

Goto build folder

Copy segmentAllImages_temporal.sh from outside

Change directory and image paths in the script 

Run it by "bash segmentAllImages_temporal.sh"


./main <data folder>/images/img_24.jpg <data folder>/images_SegnetCityS/img_24.png <data folder>/images_SegnetCityS_road/img_24.png <data folder>/images_SegnetCityS_Nonroad/img_24.png <data folder>/img_depth/img_24.png result_bin.png result_24.png


./main ../data/images/img_24.jpg ../data/images_SegnetCityS/img_24.png ../data/images_SegnetCityS_road/img_24.png ../data/images_SegnetCityS_Nonroad/img_24.png ../data/img_depth/img_24.png result_bin.png result_24.png
