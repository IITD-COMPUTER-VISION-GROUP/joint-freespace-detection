#include "graph.h"
#include "colorLines.h"
#include "tinydir.h"
// #include <opencv2/opencv.hpp>

#define K 1.5e4
#define B1 0.7
#define B2 0.7

#define S1 0.9997

using namespace std;
using namespace cv;

typedef Graph<float,float,float> GraphType;

int oneD(int x, int y , int rows){
    return x*rows + y;
}

inline void normalize(float &a , float &b){
    float temp = a + b;
    a /= temp;
    b /= temp;
}


cv::Mat get_image_of_interest(cv::Mat &image, cv::Mat &image_segnet_temp){
    cv::Mat image_segnet = cv::Mat(image.size(), image_segnet_temp.type());

    for (int i = 0; i < image_segnet.rows; ++i){
        for (int j = 0; j < image_segnet.cols; ++j){
            int x = (j*image_segnet_temp.cols)/image_segnet.cols;
            int y = (i*image_segnet_temp.rows)/image_segnet.rows;
            // std::cout << x << " " << y << " " << image_segnet_temp.rows << " " << image_segnet_temp.cols << std::endl;
            if(image_segnet_temp.at<Vec3b>(y,x)[0] == 4 && i < 1115){
                image_segnet.at<Vec3b>(i,j)[0] = 1;
                image_segnet.at<Vec3b>(i,j)[1] = 1;
                image_segnet.at<Vec3b>(i,j)[2] = 1;
            }
            else{
                image_segnet.at<Vec3b>(i,j)[0] = 0;
                image_segnet.at<Vec3b>(i,j)[1] = 0;
                image_segnet.at<Vec3b>(i,j)[2] = 0;
            }
        }
    }

    image = image.mul(image_segnet);

    return image;
}


cv::Mat get_image_for_colorline(char const * s1, char const * s2){
    tinydir_dir dir;
    int i;
    tinydir_open_sorted(&dir, s1);

    std::vector<std::string> all_images;

    for (i = 0; i < dir.n_files; i++)
    {
        tinydir_file file;
        tinydir_readfile_n(&dir, &file, i);

        // printf("%s", file.name);
        if (!file.is_dir)
        {
            all_images.push_back(file.name);
        }
        // printf("\n");
    }

    tinydir_close(&dir);


    tinydir_open_sorted(&dir, s2);

    std::vector<std::string> all_images_segnet;

    for (i = 0; i < dir.n_files; i++)
    {
        tinydir_file file;
        tinydir_readfile_n(&dir, &file, i);

        // printf("%s", file.name);
        if (!file.is_dir)
        {
            all_images_segnet.push_back(file.name);
        }
        // printf("\n");
    }

    tinydir_close(&dir);



    Mat image;
    image = imread(s1 + all_images[0], CV_LOAD_IMAGE_COLOR);

    // image = cv::Mat(image.rows*10, image.cols,image.type());
    image = cv::Mat(image.rows*all_images.size(), image.cols,image.type());

    // std::cout << image.rows << " " << image.cols << std::endl;

    // for (int i = 0; i < 10; ++i)
    for (int i = 0; i < all_images.size(); ++i)
    {

        cv::Mat temp = cv::imread(s1 + all_images[i], CV_LOAD_IMAGE_COLOR);

        cv::Mat temp_segnet = cv::imread(s2 + all_images_segnet[i], CV_LOAD_IMAGE_COLOR);

        temp = get_image_of_interest(temp, temp_segnet);

        temp.copyTo(image.rowRange(i*temp.rows, (i+1)*temp.rows).colRange(0,temp.cols));
    }

    return image;
}



cv::Mat get_single_image_for_colorline(char const * s1, char const * s2){
    // tinydir_dir dir;
    // int i;
    // tinydir_open_sorted(&dir, s1);

    std::vector<std::string> all_images;

    all_images.push_back(s1);

    // for (i = 0; i < dir.n_files; i++)
    // {
    //     tinydir_file file;
    //     tinydir_readfile_n(&dir, &file, i);

    //     // printf("%s", file.name);
    //     if (!file.is_dir)
    //     {
    //         all_images.push_back(file.name);
    //     }
    //     // printf("\n");
    // }

    // tinydir_close(&dir);


    // tinydir_open_sorted(&dir, s2);

    std::vector<std::string> all_images_segnet;

    all_images_segnet.push_back(s2);

    // for (i = 0; i < dir.n_files; i++)
    // {
    //     tinydir_file file;
    //     tinydir_readfile_n(&dir, &file, i);

    //     // printf("%s", file.name);
    //     if (!file.is_dir)
    //     {
    //         all_images_segnet.push_back(file.name);
    //     }
    //     // printf("\n");
    // }

    // tinydir_close(&dir);



    Mat image;
    image = imread(all_images[0], CV_LOAD_IMAGE_COLOR);

    // image = cv::Mat(image.rows*10, image.cols,image.type());
    image = cv::Mat(image.rows*all_images.size(), image.cols,image.type());

    // std::cout << image.rows << " " << image.cols << std::endl;

    // for (int i = 0; i < 10; ++i)
    for (int i = 0; i < all_images.size(); ++i)
    {

        cv::Mat temp = cv::imread(all_images[i], CV_LOAD_IMAGE_COLOR);

        cv::Mat temp_segnet = cv::imread(all_images_segnet[i], CV_LOAD_IMAGE_COLOR);

        temp = get_image_of_interest(temp, temp_segnet);

        temp.copyTo(image.rowRange(i*temp.rows, (i+1)*temp.rows).colRange(0,temp.cols));
    }

    return image;
}


cv::Mat reduce(cv::Mat image, cv::Size size){
    assert(image.rows >= size.height && image.cols >= size.width);
    cv::Mat ret = cv::Mat(size, image.type());
    for (int i = 0; i < ret.rows; ++i)
    {
        for (int j = 0; j < ret.cols; ++j)
        {
            int x = (j*image.cols)/ret.cols;
            int y = (i*image.rows)/ret.rows;
            ret.at<uchar>(i,j) = image.at<uchar>(y,x);
        }
    }
    return ret;
}


int main(int argc, char const *argv[])
{
    cv::Mat image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    cv::Size originalSize = image.size();

    Mat image_segnet = imread(argv[2], CV_LOAD_IMAGE_COLOR);

    // cv::resize(image_segnet,image_segnet,image.size(),INTER_NEAREST);
    cv::resize(image,image,image_segnet.size());

    Mat image_road = imread(argv[3], CV_LOAD_IMAGE_COLOR);
    cv::resize(image_road,image_road,image_segnet.size());
    // cv::resize(image_road,image_road,image.size(),INTER_NEAREST);
    cv::cvtColor(image_road, image_road, cv::COLOR_BGR2GRAY);
    
    Mat image_nonRoad = imread(argv[4], CV_LOAD_IMAGE_COLOR);
    cv::resize(image_nonRoad,image_nonRoad,image_segnet.size());
    // cv::resize(image_nonRoad,image_nonRoad,image.size(),INTER_NEAREST);
    cv::cvtColor(image_nonRoad, image_nonRoad, cv::COLOR_BGR2GRAY);

    Mat image_depth = imread(argv[5], CV_LOAD_IMAGE_COLOR);
    cv::cvtColor(image_depth, image_depth, cv::COLOR_BGR2GRAY);
    image_depth = reduce(image_depth,image_segnet.size());
    // cv::imwrite("test.png",image_depth);

    Mat image_prev = imread(argv[6], CV_LOAD_IMAGE_COLOR);
    cv::cvtColor(image_prev, image_prev, cv::COLOR_BGR2GRAY);
    image_prev = reduce(image_prev,image_segnet.size());

    const int N = image.rows*image.cols;
    Vec3b roadColor = Vec3b(4,4,4);

    GraphType *g = new GraphType(/*estimated # of nodes*/ N, /*estimated # of edges*/ N);
    for (int i = 0; i < N; ++i){
        g->add_node();
    }

    // cv:Mat large_image = get_image_for_colorline(argv[6], argv[7]);
    cv:Mat large_image = get_single_image_for_colorline(argv[1], argv[2]);
    // cv::imwrite("test.png",large_image);
    colorLines c;
    c.init(large_image,10);


    for (int i = 0; i < image.cols; ++i){
        for (int j = 0; j < image.rows; ++j){
            float p_r = 0 , p_nr = 0;
            float dp_r = 0 , dp_nr = 0;
            float dp_p_r = 0 , dp_p_nr = 0;
            
            //Segnet Probability
            float temp_p_r = (float)image_road.at<uchar>(j,i); 
            float temp_p_nr = (float)image_nonRoad.at<uchar>(j,i); 
            
            int nodeId = oneD(i,j,image.rows);
            normalize(temp_p_r ,temp_p_nr);

            p_r += temp_p_r;
            p_nr += temp_p_nr;

            //Depth Term
            uchar val = image_depth.at<uchar>(j,i);
            if (val > 0){
                dp_r = 0.9;
                dp_nr = 0.3;
            }
            else{
                dp_r = 0.0;
                dp_nr = 2.0;
            }
            // if(val > 0){
            //     dp_r = 0.9;
            //     dp_nr = 0.3;
            // }
            // else if(val == 255){
            //     dp_r = 0.0;
            //     dp_nr = 0.0;
            // }
            // else{
            //     dp_r = 0.0;
            //     dp_nr = 2.0;
            // }
            // dp_r = 0.0;
            // dp_nr = 0.0;
            dp_r *= B1;
            dp_nr *= B1;

            Vec3d pixel1= image.at<Vec3b>(j,i);
            Point3d pixel= (Point3d)pixel1;

            float dis = c.get_distance(pixel);

            // dp_r += B2*dis;

            float temp = ((B2*exp(-B2*dis)));///(2.0*PI));

            //Depth Term
            val = image_prev.at<uchar>(j,i);
            if (val > 0){
                dp_p_r = 0.9;
                dp_p_nr = 0.3;
            }
            else{
                dp_p_r = 0.0;
                dp_p_nr = 2.0;
            }

            dp_p_r *= B2;
            dp_p_nr *= B2;

            // p_r *= exp(-dp_r)*temp;
            // p_nr *= exp(-dp_nr)*(1.0-temp);

            p_r *= exp(-dp_r)*exp(-dp_p_r);
            p_nr *= exp(-dp_nr)*exp(-dp_p_r);

            // std::cout << p_r << " " << p_nr << std::endl;

            // std::cout << dis << " " << temp << std::endl;

            assert(p_r >= 0.0);
            assert(p_nr >= 0.0);

            normalize(p_r , p_nr);
            g->add_tweights( nodeId , p_r , p_nr );
        }
    }

    cv::Mat image_lap;
    // int kernel_size = 3;
    // int scale = 1;
    // int delta = 0;
    // int ddepth = -1;//CV_16S;//-1;
    // // std::cerr << "line 1" << std::endl;
    // GaussianBlur( image, image_lap, Size(3,3), 0, 0, BORDER_DEFAULT );
    // // std::cerr << "line 2" << std::endl;
    cvtColor( image, image_lap, CV_BGR2GRAY );
    // // std::cerr << "line 3" << std::endl;
    // Laplacian( image_lap, image_lap, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
    // // std::cerr << "line 4" << std::endl;
    // convertScaleAbs( image_lap, image_lap );
    // image_lap = 255 - image_lap;
    // cv::threshold(image_lap, image_lap, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    // std::cerr << "line 5" << std::endl;

    int edgeThresh = 1;
    int lowThreshold = 20;
    int const max_lowThreshold = 100;
    int ratio = 3;
    int kernel_size = 3;
    blur( image_lap, image_lap, Size(3,3) );
    Canny( image_lap, image_lap, lowThreshold, lowThreshold*ratio, kernel_size );

    image_lap = 255 - image_lap;


    // cv::imwrite("test.png",image_lap);

    int nbx[4] = {-1 , 1, 0 , 0 }; 
    int nby[4] = {0 , 0 , -1 ,1 }; 

    for (int i = 1; i < image.cols-1; i+=1){
        for (int j = 1; j < image.rows-1; j+=1){
            
            Point3d pixel1 = (Point3d)image.at<Vec3b>(j,i);  
            
            for (int k = 0; k < 4; ++k){
                Point3d pixel2 = (Point3d)image.at<Vec3b>( j+nby[k] , i+nbx[k] );
                float prob = c.get_probability2(pixel1,pixel2)*S1;
                // std::cerr << prob << " " << (((float)image_lap.at<uchar>(j,i))/255.0) << std::endl;
                prob += (1.0-S1)*(((float)image_lap.at<uchar>(j,i))/255.0);
                g->add_edge( oneD(i,j,image.rows) , oneD(i+nbx[k],j+nby[k],image.rows) , K*prob , K*prob );

                // cout<<prob<<endl;
            }
        
        }
    }

    float flow = g -> maxflow();
    printf("Flow = %f\n", flow);

    Mat result = image.clone();

    for (int i = 0; i < image.cols; ++i){
        for (int j = 0; j < image.rows; ++j){
            
            int n1;
            n1 = oneD(i,j,image.rows);
            
            if (g->what_segment(n1) == GraphType::SOURCE){
                    // result.at<Vec3b>(j,i) = Vec3b(255,105,180);
                    result.at<Vec3b>(j,i) = Vec3b(255,255,255);
                }
            else{
                    result.at<Vec3b>(j,i) = Vec3b(0,0,0);
                }// cout<<"SINK"<<endl;
            
        }
    }
    imwrite( argv[7], result );
    for (int i = 0; i < image.cols; ++i){
        for (int j = 0; j < image.rows; ++j){
            
            int n1;
            n1 = oneD(i,j,image.rows);
            
            if (g->what_segment(n1) == GraphType::SOURCE){
                    result.at<Vec3b>(j,i) = Vec3b(255,105,180);
                }
            else{
                    result.at<Vec3b>(j,i) = Vec3b(0,0,0);
                }// cout<<"SINK"<<endl;
            
        }
    }
    // imwrite( "data/segmentation_wi.png", image );
    Mat dst;
    float alpha , beta;
    alpha = 0.8;
    // beta = ( 1.0 - alpha );
    beta = 0.6;
    // cv::resize(image,image,img.size());

    addWeighted( image, alpha, result, beta, 0.0, dst);
    imwrite( argv[8], dst );
    
  
    

    return 0;
}
