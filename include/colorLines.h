#include "point.h"
#include <vector>
#include <math.h>
#include <string>
#include <opencv2/opencv.hpp>

#define PI 3.14159265
#define NB_K 5

using namespace cv;
using namespace std;

typedef std::vector<point> colorLine;

class colorLines
{

public:
	
	std::vector<colorLine> lines;
	
	// to be used only after cleanup
	// cleanup called in init
	std::vector< std::vector<colorLine> > lines_cleanedup;
	int radius;
	
	colorLines();
	~colorLines();

	void init(cv::Mat img , const int r);

	void cleanup(colorLine line);
	
	std::vector<float> get_probability(cv::Point3d input);

	float get_distance(cv::Point3d input);

	float get_probability2(Point3d pt1 , Point3d pt2);
};