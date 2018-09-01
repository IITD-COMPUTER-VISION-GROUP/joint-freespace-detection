#include "colorLines.h"

colorLines::colorLines(){}
colorLines::~colorLines(){}


cv::Mat low_pass(cv::Mat src)
{
	Point anchor = Point( -1, -1 );
  	double delta = 0;
  	int ddepth = -1;
	int kernel_size = 3;
	Mat kernel = Mat::ones( kernel_size, kernel_size, CV_32F )/ (float)(kernel_size*kernel_size);
	Mat dst;
    /// Apply filter
	filter2D(src, dst, ddepth , kernel, anchor, delta, BORDER_DEFAULT );
	// bilateralFilter( src, dst, 15, 40, 40 );
	return dst;
}


std::vector<cv::Point2f> local_maxima(cv::Mat src)
{
	const int border = 2;
	std::vector<cv::Point2f> ret;

	int nbx[8] = {-1 , -1, -1, 0 ,0 , 1 , 1, 1}; 
	int nby[8] = {-1 , 0 , 1 , 1 ,-1, -1, 0, 1}; 
	
	for(int y = border; y < src.rows - border; y++){
		for(int x = border; x < src.cols - border; x++){
			bool maxima = true;
			for (int k = 0; k < 8; ++k){
				if ( src.at<float>(y+nby[k], x+nbx[k]) >= src.at<float>(y, x) ){
					maxima = false;
					break;
				}
			}

			if(maxima){
				ret.push_back(cv::Point2f((float)x,(float)y));
			}
		}
	}
	return ret;
}

cv::Mat affiliation(cv::Mat src, std::vector<cv::Point2f> maximas)
{
	cv::Mat ret = src.clone();
	for(int y = 0; y < src.rows; y++){
		for(int x = 0; x < src.cols; x++){
			float mindist = ((float)x-maximas[0].x)*((float)x-maximas[0].x) + ((float)y-maximas[0].y)*((float)y-maximas[0].y);
			ret.at<float>(y,x) = 0.0;
			for (int i = 1; i < maximas.size(); ++i)
			{
				float temp = ((float)x-maximas[i].x)*((float)x-maximas[i].x) + ((float)y-maximas[i].y)*((float)y-maximas[i].y);
				if(temp < mindist){
					mindist = temp;
					ret.at<float>(y,x) = float(i);
				}
			}
		}
	}
	return ret;
}

std::vector<cv::Point3f> calc_gaussians(cv::Mat hist, cv::Mat belongs, std::vector<cv::Point2f> &maximas)
{
	std::vector<cv::Point3f> ret;
	std::vector<float> count;
	for (int i = 0; i < maximas.size(); ++i){
		ret.push_back(cv::Point3f(0.0,0.0,0.0));
		count.push_back(0.0);
	}

	for (int y = 0; y < hist.rows; ++y){
		for (int x = 0; x < hist.cols; ++x){
			int where = (int)belongs.at<float>(y,x);
			ret[where].x += hist.at<float>(y,x)*(float)x;
			ret[where].y += hist.at<float>(y,x)*(float)y;
			ret[where].z += hist.at<float>(y,x)*(float)x*(float)x + hist.at<float>(y,x)*(float)y*(float)y;
			count[where] += hist.at<float>(y,x);
			// if(hist.at<float>(y,x) != 0.0)
			// 	std::cout << where << " " << hist.at<float>(y,x) << " " << ret[where].x << " " << ret[where].y << " " << count[where] << std::endl;
		}
	}

	std::vector<cv::Point2f> temp = maximas;
	maximas.clear();
	std::vector<cv::Point3f> rt;
	for (int i = 0; i < temp.size(); ++i)
	{
		if(count[i] > 0.0)
		{
			ret[i].x /= count[i];
			ret[i].y /= count[i];
			ret[i].z /= count[i];
			ret[i].z -= (ret[i].x*ret[i].x + ret[i].y*ret[i].y);
			// temp.push_back(maximas[i]);
			maximas.push_back(temp[i]);
			rt.push_back(ret[i]);
		}
	}
	// for (int j = 0; j < ret.size(); ++j){
	// 	std::cout << j << " " << (int)ret[j].y << " " << ret[j].y << " " << (int)ret[j].x << " " << ret[j].x << std::endl;
	// }
	return rt;
}

bool checkNbrs(Point p, Mat maximaPos , Mat maximaValues , const int search_size)
{
	for (int x = p.x - search_size; x < p.x + search_size; ++x){
		for (int y = p.y - search_size; y < p.y + search_size; ++y){
			if (x >= 0 && y >= 0 && x < maximaPos.cols && y < maximaPos.rows && maximaPos.at<uchar>(y,x) > 0 ){
				if ( maximaValues.at<float>(y,x) > maximaValues.at<float>( p.y , p.x ) )
				{
					return true;
				}
			}
		}
	}

	return false;
}


vector<cv::Point2f> combineMaximas(vector<cv::Point2f> maximas , Mat maximaValues , const int search_size)
{
	
	vector<cv::Point2f> maximas_combined;	
	Mat maximaPos = Mat::zeros(360,360,CV_8UC1);
	for (int i = 0; i < maximas.size(); ++i){
		maximaPos.at<uchar>((int)maximas[i].y , (int)maximas[i].x) = 255;
	}

	for (int i = 0; i < maximas.size(); ++i){
		Point2i p = Point2i( (int)maximas[i].x , (int)maximas[i].y );

		if ( ! checkNbrs(p,maximaPos,maximaValues,search_size) ){
			maximas_combined.push_back( maximas[i] );
		}
	}

	return maximas_combined;
}


int get_nearest_index(cv::Point3f pt, std::vector<cv::Point3f> &gaussians){
	int ret = 0;
	float min_distance = (pt.x-gaussians[0].x)*(pt.x-gaussians[0].x) + (pt.y-gaussians[0].y)*(pt.y-gaussians[0].y);
	for (int i = 0; i < gaussians.size(); ++i){
		float temp_distance = (pt.x-gaussians[i].x)*(pt.x-gaussians[i].x) + (pt.y-gaussians[i].y)*(pt.y-gaussians[i].y);
		// std::cout << temp_distance << " " << min_distance << std::endl;
		if(temp_distance  < min_distance){
			min_distance = temp_distance;
			ret = i;
		}
	}
	// std::cout << std::endl;
	return ret;
}


void formLinesLeft(std::vector< std::vector<cv::Point3f> > &histogram_slices_gaussians , std::vector< std::vector<cv::Point2f> > &histogram_slices_maximas , std::vector<cv::Mat> &histogram_slices , const int search_size, std::vector<colorLine> &lines, int number_lines, int start_point, std::vector<int> affiliated_color_line, const int r){
	//do something
	// assert(histogram_slices_maximas[start_point].size() == affiliated_color_line.size());
	// std::cout << "left " << start_point << std::endl;
	if(start_point == 0){
		return;
	}
	--start_point;
	{
		std::vector<int> acl_new;
		for (int i = 0; i < histogram_slices_gaussians[start_point].size(); ++i){
			int temp_index = start_point + 1;
			while(histogram_slices_gaussians[temp_index].size() == 0){
				++temp_index;
			}
			int nearest_index = affiliated_color_line[get_nearest_index(histogram_slices_gaussians[start_point][i] , histogram_slices_gaussians[temp_index])];
			acl_new.push_back(nearest_index);

			point temp;
			float theta = histogram_slices_maximas[start_point][i].x*PI/180.0;
			float phi = histogram_slices_maximas[start_point][i].y*PI/180.0;
			temp.b = r*start_point*sin(phi)*cos(theta);
			temp.g = r*start_point*sin(phi)*sin(theta);
			temp.r = r*start_point*cos(phi);
			temp.mu_x = histogram_slices_gaussians[start_point][i].x;
			temp.mu_y = histogram_slices_gaussians[start_point][i].y;
			temp.sigma = histogram_slices_gaussians[start_point][i].z;
			lines[nearest_index].push_back(temp);
		}
		if(histogram_slices_gaussians[start_point].size() == 0){
			formLinesLeft(histogram_slices_gaussians, histogram_slices_maximas , histogram_slices , 5, lines, number_lines, start_point, affiliated_color_line, r);
		}
		else{
			formLinesLeft(histogram_slices_gaussians, histogram_slices_maximas , histogram_slices , 5, lines, number_lines, start_point, acl_new, r);
		}
	}

}

void formLinesRight(std::vector< std::vector<cv::Point3f> > &histogram_slices_gaussians , std::vector< std::vector<cv::Point2f> > &histogram_slices_maximas , std::vector<cv::Mat> &histogram_slices , const int search_size, std::vector<colorLine> &lines, int number_lines, int start_point, std::vector<int> affiliated_color_line, const int r){
	//do something
	// std::cout << "right " << start_point << std::endl;
	// assert(histogram_slices_maximas[start_point].size() == affiliated_color_line.size());
	const int num_bins = (450/r) + 1;
	assert(num_bins == histogram_slices.size());
	// std::cout << "right " << start_point << " " << num_bins - 1 << std::endl;
	if(start_point == num_bins - 1){
		return;
	}
	++start_point;
	{
		std::vector<int> acl_new;
		// std::cout << histogram_slices_gaussians[start_point].size() << std::endl;
		for (int i = 0; i < histogram_slices_gaussians[start_point].size(); ++i){
			int temp_index = start_point - 1;
			while(histogram_slices_gaussians[temp_index].size() == 0){
				--temp_index;
			}
			int nearest_index = affiliated_color_line[get_nearest_index(histogram_slices_gaussians[start_point][i] , histogram_slices_gaussians[temp_index])];
			acl_new.push_back(nearest_index);


			point temp;
			float theta = histogram_slices_maximas[start_point][i].x*PI/180.0;
			float phi = histogram_slices_maximas[start_point][i].y*PI/180.0;
			temp.b = r*start_point*sin(phi)*cos(theta);
			temp.g = r*start_point*sin(phi)*sin(theta);
			temp.r = r*start_point*cos(phi);
			temp.mu_x = histogram_slices_gaussians[start_point][i].x;
			temp.mu_y = histogram_slices_gaussians[start_point][i].y;
			temp.sigma = histogram_slices_gaussians[start_point][i].z;
			lines[nearest_index].push_back(temp);
		}
		// int a;
		// std::cin >> a;
		if(histogram_slices_gaussians[start_point].size() == 0){
			formLinesRight(histogram_slices_gaussians, histogram_slices_maximas , histogram_slices , 5, lines, number_lines, start_point, affiliated_color_line, r);
		}
		else{
			formLinesRight(histogram_slices_gaussians, histogram_slices_maximas , histogram_slices , 5, lines, number_lines, start_point, acl_new, r);
		}
	}
}

// #include <stdlib.h>
void colorLines::init(cv::Mat img,  const int r)
{
	this->radius = r;
	const int num_bins = (450/r) + 1;
	std::vector<vector<Point>> imagePts;
	imagePts.resize(num_bins);


	// Mat hist = Mat::zeros(360,360,CV_32FC1);
	Point3d origin =  Point3d(0,0,0);
	Point3d eps =  Point3d(0.01,0.01,0.01);

	for (int i = 0; i < img.cols; ++i){
		for (int j = 0; j < img.rows; ++j){
			
			Vec3d pixel1= img.at<Vec3b>(j,i);
			Point3d pixel= (Point3d)pixel1;
			
			// cout<<i<<"   "<<j<<"  "<<pixel<<endl;
			Point3d direction_vec = pixel - origin;

			double magnitude = norm(direction_vec);
			direction_vec.x /= magnitude;
			direction_vec.y /= magnitude;
			direction_vec.z /= magnitude;

			magnitude = norm(pixel);
			
			if ( magnitude > 0 ){	
				int bin_id = magnitude/r + 1; 
				Point3d proj_pt = bin_id*direction_vec + eps;
	
				double theta = atan(proj_pt.y/proj_pt.x);
	
				double temp = sqrt(proj_pt.x*proj_pt.x + proj_pt.y*proj_pt.y);
				double phi = atan(  temp/proj_pt.z );
	
				int x = (int)(theta*180/PI);
				int y = (int)(phi*180/PI);

				// hist.at<float>(y,x) += 1;
				imagePts[bin_id].push_back( Point(x,y) );
				// cout<<hist.at<float>(y,x)<<endl;
			}
		}
	}

	std::vector<cv::Mat> histogram_slices;
	std::vector< std::vector<cv::Point2f> > histogram_slices_maximas;
	std::vector< std::vector<cv::Point3f> > histogram_slices_gaussians;
	std::vector<int> histogram_slice_size;
	int max_lines = 0;
	int max_lines_index = 0;

	for (int i = 0; i < num_bins; ++i)
	{
		// cout<<imagePts[i].size()<<endl;
		Mat hist = Mat::zeros(360,360,CV_32FC1);

		for (int j = 0; j < imagePts[i].size(); ++j){
			hist.at<float>(imagePts[i][j].y , imagePts[i][j].x ) += 1;		
		}

		//Call your gaussian fitting function on hist
		// imwrite("hist"+to_string(i)+".png",hist);
		cv::Mat dst = low_pass(hist);
		Mat ucharMatScaled;
		dst.convertTo(ucharMatScaled, CV_8UC1, 255, 0); 
		// imwrite("dstScaled"+to_string(i)+".png",ucharMatScaled);
		// imwrite("dst"+to_string(i)+".png",dst);

		std::vector<cv::Point2f> maximas = local_maxima(dst);

		Mat temp = Mat::zeros(360,360,CV_8UC1);
		for (int j = 0; j < maximas.size(); ++j){
			temp.at<uchar>((int)maximas[j].y , (int)maximas[j].x) = 255;
		}


		// imwrite("temp"+to_string(i)+".png",temp);
		// cout<<"Here "<< maximas.size() <<endl;


		std::vector<cv::Point2f> maximas_combined = combineMaximas(maximas , dst ,5);
		temp = Mat::zeros(360,360,CV_8UC1);
		for (int j = 0; j < maximas_combined.size(); ++j){
			temp.at<uchar>((int)maximas_combined[j].y , (int)maximas_combined[j].x) = 255;
		}


		if(i < 10){
			// imwrite("tempC0"+to_string(i)+".png",temp);
		}
		else{
			// imwrite("tempC"+to_string(i)+".png",temp);
		}
		// cout<<"HereC "<< maximas_combined.size() <<endl;


		// assert(maximas.size() > 0);
		if(maximas.size() > 0){
			cv::Mat belongs = affiliation(hist, maximas);
			std::vector<cv::Point3f> gaussians = calc_gaussians(hist,belongs,maximas);
			std::vector<cv::Point2f> stay_put;
			
			Mat result = Mat::zeros(360,360,CV_8UC1);
			for (int j = 0; j < gaussians.size(); ++j){
				result.at<uchar>((int)gaussians[j].y , (int)gaussians[j].x) = (uchar)((int)gaussians[j].z);
				stay_put.push_back(cv::Point2f(gaussians[j].x,gaussians[j].y));
			}

			if(gaussians.size() > max_lines){
				max_lines = gaussians.size();
				max_lines_index = i;
			}
			
			histogram_slices.push_back(result);
			histogram_slice_size.push_back(gaussians.size());
			histogram_slices_gaussians.push_back(gaussians);
			histogram_slices_maximas.push_back(stay_put);
			result.convertTo(ucharMatScaled, CV_8UC1, 255, 0); 
			// imwrite("resultScaled"+to_string(i)+".png",ucharMatScaled);

		}
		else{
			Mat result = Mat::zeros(360,360,CV_8UC1);
			histogram_slices.push_back(result);
			histogram_slice_size.push_back(0);
			std::vector<cv::Point3f> gaussians;
			histogram_slices_gaussians.push_back(gaussians);
			std::vector<cv::Point2f> stay_put;
			histogram_slices_maximas.push_back(stay_put);
		}
		


		//gaussians in std::vector<cv::Point3f> gaussians
		// each Point3f represents a gaussian
		// x is mean in x and
		// y is mean in y and
		// z is variance

	}

	// all histogram slices present here
	// std::vector<colorLine> lines;
	// for (int i = 0; i < histogram_slices_maximas.size(); ++i)
	// {
	// 	std::cout << histogram_slices_maximas[i].size() << std::endl;
	// }
	// std::cout << histogram_slices_maximas[max_lines_index].size() << " " << max_lines << " " << histogram_slices_gaussians[max_lines_index].size() << std::endl;
	assert(histogram_slices_maximas[max_lines_index].size() == max_lines);
	
	//initialize the color lines with some points
	std::vector<int> affiliated_color_line;
	// for (int l = 0; l < 2; ++l)
	for (int l = 0; l < histogram_slices_maximas[max_lines_index].size(); ++l)
	{
		std::vector<point> temp_vec;
		point temp;
		float theta = histogram_slices_maximas[max_lines_index][l].x*PI/180.0;
		float phi = histogram_slices_maximas[max_lines_index][l].y*PI/180.0;
		temp.b = r*max_lines_index*sin(phi)*cos(theta);
		temp.g = r*max_lines_index*sin(phi)*sin(theta);
		temp.r = r*max_lines_index*cos(phi);
		temp.mu_x = histogram_slices_gaussians[max_lines_index][l].x;
		temp.mu_y = histogram_slices_gaussians[max_lines_index][l].y;
		temp.sigma = histogram_slices_gaussians[max_lines_index][l].z;
		temp_vec.push_back(temp);
		lines.push_back(temp_vec);
		affiliated_color_line.push_back(l);
	}

	// std::cerr << lines.size() << std::endl;
	formLinesLeft(histogram_slices_gaussians, histogram_slices_maximas , histogram_slices , 5, lines, max_lines, max_lines_index, affiliated_color_line, r);
	// std::cerr << lines.size() << std::endl;
	formLinesRight(histogram_slices_gaussians, histogram_slices_maximas , histogram_slices , 5, lines, max_lines, max_lines_index, affiliated_color_line, r);
	

	// std::cerr << lines.size() << std::endl;
	int max_size_index = 0;
	int max_size = lines[0].size();
	for (int i = 0; i < lines.size(); ++i){
		// cleanup(lines[i]);
		// std::cerr << lines[i].size() << std::endl;
		// std::cout << lines[i].size() << std::endl;
		if(max_size < lines[i].size()){
			max_size = lines[i].size();
			max_size_index = i;
		}
		// for (int j = 0; j < lines[i].size(); ++j)
		// {
		// 	std::cerr << lines[i][j].r << "\n" << lines[i][j].g << "\n" << lines[i][j].b << std::endl;
		// }
	}

	std::vector<colorLine> temp;
	temp.push_back(lines[max_size_index]);
	lines = temp;

	// std::cerr << lines.size() << std::endl;
	for (int i = 0; i < lines.size(); ++i){
		cleanup(lines[i]);
		// std::cerr << lines[i].size() << std::endl;
		// std::cout << lines[i].size() << std::endl;
		/*for (int j = 0; j < lines[i].size(); ++j)
		{
			std::cerr << lines[i][j].r << "\n" << lines[i][j].g << "\n" << lines[i][j].b << std::endl;
		}*/
	}

	// std::cout << lines[0].size() << std::endl;
	// std::cout << lines.size() << " " << lines_cleanedup.size() << std::endl;

	// Mat ret = Mat::zeros(img.size(),CV_8UC1);
	// for (int i = 0; i < ret.cols; ++i){

	// 	for (int j = 0; j < ret.rows; ++j){

	// 		// std::cout << "start loop " << i << " " << j << std::endl;
	// 		Vec3d pixel1= img.at<Vec3b>(j,i);
	// 		Point3d pixel= (Point3d)pixel1;
	// 		// std::cout << "start get probability" << std::endl;
	// 		std::vector<float> probs = get_probability(pixel);
	// 		// std::vector<float> probs;
	// 		// probs.resize(10);
	// 		// std::cout << "end get probability" << std::endl;
	// 		int max_id = 0;
	// 		float max_prob = probs[0];
	// 		// std::cout << probs[0] << std::endl;
	// 		for (int k = 0; k < probs.size(); ++k)
	// 		{
	// 			// std::cout << probs[k] << " " << k << " " << probs.size() << std::endl;
	// 			if(probs[k] > max_prob){
	// 				max_id = k;
	// 				max_prob = probs[k];
	// 			}
	// 		}
	// 		// std::cerr << max_id << std::endl;
	// 		// std::cout << "end loop " << i << " " << j << std::endl;
	// 		ret.at<uchar>(j,i) = max_id + 1;
	// 	}

	// 	// std::cout << " temp " << std::endl;
	// }
	// cv::Mat retMatScaled;
	// ret.convertTo(retMatScaled, CV_8UC1, 255, 0);
	// imwrite("color_line_id.png",ret);

}


void colorLines::cleanup(colorLine line){
	int r = radius;
	const int num_bins = (450/r) + 1;
	std::vector<colorLine> temp;
	lines_cleanedup.push_back(temp);
	lines_cleanedup[lines_cleanedup.size()-1].resize(num_bins);
	for (int i = 0; i < line.size(); ++i){
		float magnitude = norm(cv::Point3d(line[i].b,line[i].g,line[i].r));
		int bin_id = magnitude/r + 1;
		lines_cleanedup[lines_cleanedup.size()-1][bin_id].push_back(line[i]);
	}
}

inline float gaussian_probability(float x, float y, float mu_x, float mu_y, float sigma){
	float ret = 1/(2.0*PI*sigma*sigma);
	float temp = -((x-mu_x)*(x-mu_x) + (y-mu_y)*(y-mu_y))/(2.0*sigma*sigma);
	ret *= exp(temp);
	// cout<<"Prob : "<<ret<<endl;
	return ret;
}

std::vector<float> colorLines::get_probability(cv::Point3d input){
	int r = radius;
	float magnitude = norm(input);
	int bin_id = magnitude/r + 1;
	const int num_bins = (450/r) + 1;
	std::vector<float> ret;
	ret.resize(lines_cleanedup.size());
	Point3d eps =  Point3d(0.01,0.01,0.01);
	for (int i = 0; i < lines_cleanedup.size(); ++i){
		ret[i] = 0.0;
		assert(num_bins == lines_cleanedup[i].size());
		if(lines_cleanedup[i].size() > 0){
			// for (int j = 0; j < num_bins; ++j)
			// {
			assert(bin_id < num_bins);
			int j = bin_id;//just for fun
			if(lines_cleanedup[i][j].size() > 0){
				// float max_prob = 0.0;
				for (int k = 0; k < lines_cleanedup[i][j].size(); ++k){
					Point3d proj_pt = (double)bin_id*(input) + eps;
		
					double theta = atan(proj_pt.y/proj_pt.x);
		
					double temp = sqrt(proj_pt.x*proj_pt.x + proj_pt.y*proj_pt.y);
					double phi = atan(  temp/proj_pt.z );
		
					float x = (theta*180/PI);
					float y = (phi*180/PI);
					float temp_prob = gaussian_probability(x, y, lines_cleanedup[i][j][k].mu_x, lines_cleanedup[i][j][k].mu_y, lines_cleanedup[i][j][k].sigma);
					// if(temp_prob > ret[i]){
					ret[i] = max(temp_prob,ret[i]);
					// }
				}
			}
			// }
		}
	}
	return ret;
}

inline float distance_from_gaussian(float x, float y, float mu_x, float mu_y, float sigma){
	float ret = ((x-mu_x)*(x-mu_x) + (y-mu_y)*(y-mu_y))/(2.0*sigma*sigma);
	// cout<<"Prob : "<<ret<<endl;
	return ret;
}

float colorLines::get_distance(cv::Point3d input){
	int r = radius;
	float magnitude = norm(input);
	int bin_id = magnitude/r + 1;
	const int num_bins = (450/r) + 1;
	std::vector<float> ret;
	ret.resize(lines_cleanedup.size());
	Point3d eps =  Point3d(0.01,0.01,0.01);
	for (int i = 0; i < lines_cleanedup.size(); ++i){
		ret[i] = 0.0;
		assert(num_bins == lines_cleanedup[i].size());
		if(lines_cleanedup[i].size() > 0){
			// for (int j = 0; j < num_bins; ++j)
			// {
			assert(bin_id < num_bins);
			int j = bin_id;//just for fun
			if(lines_cleanedup[i][j].size() > 0){
				// float max_prob = 0.0;
				for (int k = 0; k < lines_cleanedup[i][j].size(); ++k){
					Point3d proj_pt = (double)bin_id*(input) + eps;
		
					double theta = atan(proj_pt.y/proj_pt.x);
		
					double temp = sqrt(proj_pt.x*proj_pt.x + proj_pt.y*proj_pt.y);
					double phi = atan(  temp/proj_pt.z );
		
					float x = (theta*180/PI);
					float y = (phi*180/PI);
					float temp_prob = distance_from_gaussian(x, y, lines_cleanedup[i][j][k].mu_x, lines_cleanedup[i][j][k].mu_y, lines_cleanedup[i][j][k].sigma);
					// if(temp_prob > ret[i]){
					ret[i] = max(temp_prob,ret[i]);
					// }
				}
			}
			// }
		}
	}
	float min_distance = ret[0];
	for (int i = 0; i < ret.size(); ++i)
	{
		min_distance = min(min_distance,ret[i]);
	}
	return min_distance;
}


float colorLines::get_probability2(Point3d pt1 , Point3d pt2){

	std::vector<float> probs1 = get_probability(pt1);
	std::vector<float> probs2 = get_probability(pt2);

	float prob = 0;

	for (int i = 0; i < probs1.size(); ++i){
		prob += probs1[i]*probs2[i];
	}

	return prob;
}
