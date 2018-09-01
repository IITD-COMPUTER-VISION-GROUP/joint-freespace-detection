#!/bin/sh

direc=`pwd`
fullpath="/media/netra/Data2/continental_videos/SpeedBreaker&Potholes/potholes/p5"
fullpathTemp="/media/netra/Data2/continental_videos/SpeedBreaker&Potholes/potholes/p5/images"
temp=".png"
t1="1"
t2="1"

for i in $(ls $fullpathTemp | sort -V); do
	# echo $i
	# echo ${i/jpg/png}
	# echo $i$temp
	# if [[ $t1 == $t2 ]]; then
	# time ./main $fullpath/images/$i $fullpath/images_SegnetCityS/${i/jpg/png} $fullpath/images_SegnetCityS_road/${i/jpg/png} $fullpath/images_SegnetCityS_Nonroad/${i/jpg/png} $fullpath/images_Depth/$i$temp $fullpath/results_bin_segnet/${i/jpg/png} $fullpath/results_segnet/${i/jpg/png}
	time ./main $fullpath/images/$i $fullpath/images_SegnetCityS/${i/jpg/png} $fullpath/images_SegnetCityS_road/${i/jpg/png} $fullpath/images_SegnetCityS_Nonroad/${i/jpg/png} $fullpath/images_Depth/${i/jpg/png} $fullpath/results_bin/${i/jpg/png} $fullpath/results/${i/jpg/png}
	#	t1="0"
	#	prev=$i
	# fi
	# ./binary $fullpath/images/$i $fullpath/images_SegnetCityS/${i/jpg/png} $fullpath/images_SegnetCityS_road/${i/jpg/png} $fullpath/images_SegnetCityS_Nonroad/${i/jpg/png} $fullpath/images_Depth/$i$temp $fullpath/results_bin/${prev/jpg/png} $fullpath/results_bin/${i/jpg/png} $fullpath/results_temporal/${i/jpg/png}
	prev=$i
done
