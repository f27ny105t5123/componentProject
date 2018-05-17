#ifndef ROTATEIMAGE_HPP
#define ROTATEIMAGE_HPP

#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

void angleCalibration(Mat &cannyImage, double angle);
double getAngle(const Mat &img, int thresholdVal);

#endif
