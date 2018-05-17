#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

#include "rotateImage.hpp"

double getAngle(const Mat &img, int thresholdVal)
{
	Mat img_cny;
	threshold(img, img_cny, thresholdVal, 255, THRESH_BINARY_INV);
	namedWindow("rotate", WINDOW_NORMAL);
	imshow("rotate", img_cny);
	vector<vector<Point>> contours;
	findContours(img_cny, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	
	RotatedRect maxrect;
	double maxlength = 0;
	for (int j = 0; j < contours.size(); j++)
	{
		RotatedRect nrect = minAreaRect(contours[j]);
		if ((nrect.size.height + nrect.size.width)>maxlength)
		{
			maxrect = minAreaRect(contours[j]);
			maxlength = nrect.size.height + nrect.size.width;
		}
	}
	Point2f topoint[4];
	maxrect.points(topoint); 
	/*
	Mat img_color;
	cvtColor(img_cny, img_color, CV_GRAY2BGR);
	for (int i = 0; i < 4; ++i)
		line(img_color, topoint[i], topoint[(i + 1) % 4], Scalar(0, 255, 255));
	namedWindow("anglecolor", WINDOW_NORMAL);
	imshow("anglecolor", img_color);
	cout << "angle is " << maxrect.angle << endl;
	*/
	return maxrect.angle;
}

void angleCalibration(Mat &cannyImage, double angle)	//function interface
{	
	if(angle < -45)
	{
		angle = 90 + angle;
	}
	warpAffine(cannyImage, cannyImage, getRotationMatrix2D(Point2f(cannyImage.cols/2., cannyImage.rows/2.), angle, 1.0), cannyImage.size());
}
