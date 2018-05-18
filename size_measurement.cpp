#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string> 
#include "rotateImage.hpp"
#include "crackDetect.hpp"

using namespace std;
using namespace cv;
 
void top_measure(Mat &srcImage, bool sizeFlag, bool deformationFlag, int *sidePos, double angle);
void internal_measure(const Mat &srcImage, double angle);
void crackDetector(Mat &srcImage, int *sidePos, double angle);

int main()
{
	clock_t begTime = clock();
	Mat srcImage = imread("../photo/1.bmp", IMREAD_GRAYSCALE);
	if(srcImage.empty())
	{	
		cout << "Loading image is a failure." << endl;
		exit(-1);
	}
	int mode = 2;
	
	int sidePos[4]; 
	double angle;
	switch(mode)
	{
		//顶面检测
		case 0:
			top_measure(srcImage, true, true, sidePos, getAngle(srcImage, 110));
		        break;

		//侧面检测
		case 1:
			angle = getAngle(srcImage, 80);
			internal_measure(srcImage, angle);
		        break;
				
		//矩形侧面裂缝检测
		case 2:
			//angle = getAngle(srcImage, 50);
			//top_measure(srcImage, true, false, sidePos, angle);
			crackDetector(srcImage, sidePos, angle);
			break;
		default:
			cerr << "Mode number error!\n";break;
	}
	cout << "The program running duration is " << static_cast<double>(clock() - begTime) / CLOCKS_PER_SEC << " sec" << endl;
	waitKey(0);
	return EXIT_SUCCESS;
}

void crackDetector(Mat &srcImage, int *sidePos, double angle)
{
	Mat tempImage;
	srcImage.copyTo(tempImage);
	angleCalibration(tempImage, angle);
	Mat crackROI = (tempImage.colRange(sidePos[0] + 30, sidePos[1] - 30)).rowRange(sidePos[2] + 10, sidePos[3] - 30);
	detector(crackROI);
}


//rotateImage流程应该可以进一步优化

/*
 * 该函数测量适用于“山”字形的侧面图
 * 零件开口基本朝左（角度不要太大，程序会自动纠正微小的倾斜角度）
 * 保证远景与前景下上错开
 * 保证边缘没有影响canny检测的缺损（先缺损检测）
 * 须确定零件顶面的外轮廓没有形变
 * 零件图不要有分界明显的阴影
 * 全图不要有影响canny检测的污点！
 */
//该面不测量零件高度。高度方向要求的视差会导致测量不准确，但是应该测量两头的高度形变，确认没有形变再从另一侧面测量实际值
//注：两头斜坡样的形变暂时无法分辨，因为透视的关系，不管在哪一个面看到的都是斜坡形变的最高部分。
#define PARALLAX_THRESHOLD (edgePoint.size()*0.012) 	//或者固定值13。视差阈值，一列像素的数量小于该值的部分会被当做视差剔除
#define NOISE_THRESHOLD 5				//一列像素的数量小于该值，认为是噪声，剔除
void internal_measure(const Mat &srcImage, double angle)	//侧面内部尺寸对称+圆柱直径检测
{
	clock_t beg = clock();
	Mat cannyImage;
	Canny(srcImage, cannyImage, 100, 100, 3, true);
	angleCalibration(cannyImage, angle);

	vector<Point2i> edgePoint;	//换成list没有发现效率提高，反而因为list不支持随机访问导致代码复杂化
	edgePoint.reserve(1024);
	map<int, int> statistics;
	for(auto rowCounter = 0; rowCounter < cannyImage.rows; ++rowCounter)	//扫描内部边缘
	{
		unsigned char* rowPointer =  cannyImage.ptr<unsigned char>(rowCounter);
		for(auto colCounter = 0; colCounter < cannyImage.cols; ++colCounter)
		{
			if(*(rowPointer + colCounter)	    && 
			   (cannyImage.cols - colCounter) > 10)	//边缘不应在图像底部
			{
				edgePoint.push_back(Point2i(rowCounter, colCounter));	//找到本行第一个边缘点，换行继续寻找
				++statistics[colCounter];
				break;
			}
		}
	}
	//初步确定flagPoint，后面再进行修正
	for(auto tmpPtr = statistics.begin(); tmpPtr != statistics.end();)
	{
		if(tmpPtr->second < NOISE_THRESHOLD) 			//一列的像素数小于阈值，剔除噪声像素
			tmpPtr = statistics.erase(tmpPtr);	
		else 
			++tmpPtr;
	}
	for(auto tmpPtr = statistics.begin(); tmpPtr != statistics.end(); ++tmpPtr)
		cout << tmpPtr->first << ":" << tmpPtr->second << endl;
	for(auto ptr = edgePoint.begin(); ptr != edgePoint.end();)
	{
		if(statistics.find(ptr->y) == statistics.end())
			ptr = edgePoint.erase(ptr);
		else
			++ptr;
	}
	//下面根据坐标差距大于某个阈值这个条件，即可确定各个边缘并测量距离
	vector<Point2i> flagPoint;
	flagPoint.reserve(4);	//内部边缘的标记点，“山”形的零件应有4个标记点
	for(auto tmpPointer = edgePoint.begin(); tmpPointer != edgePoint.end() - 1; ++tmpPointer)
	{
		if(abs(tmpPointer->y - (tmpPointer + 1)->y) > 30)	//高度差达到阈值，认为是flagPoint
		{
			flagPoint.push_back(tmpPointer->y < (tmpPointer + 1)->y ? *tmpPointer : *(tmpPointer + 1));	//每次都采集前景的像素点（偏高的点）
		}
	}
	//后景透视到前景的部分剔除
	statistics.clear();
	for(auto ptr = edgePoint.begin(); *ptr != flagPoint[0]; ++ptr)
	{
		++statistics[ptr->y];
	}
	cout << "edgePoint.size = " << edgePoint.size() << endl;
	for(auto tmpPtr = statistics.begin(); tmpPtr != statistics.end();)
	{
		if(tmpPtr->second < PARALLAX_THRESHOLD) 			//一列的像素数小于阈值，剔除视差像素
			tmpPtr = statistics.erase(tmpPtr);	
		else 
			++tmpPtr;
	}
	for(auto ptr = edgePoint.begin(); ;)		//对第一个flagPoint之前的像素点重新做统计，剔除视差部分
	{
		if(statistics.find(ptr->y) == statistics.end())
			ptr = edgePoint.erase(ptr);
		else
			++ptr;
		if(*ptr == flagPoint[0])
		{
			edgePoint.erase(ptr);
			break;
		}
	}
	for(auto ptr = edgePoint.begin(); ptr != edgePoint.end(); ++ptr)
	{
		if(ptr->x > flagPoint[0].x)
		{
			flagPoint[0] = *(ptr - 1);
			break;
		}
	}
	statistics.clear();
	for(auto ptr = edgePoint.end() - 1; *ptr != flagPoint[flagPoint.size() - 1]; --ptr)
	{
		++statistics[ptr->y];
	}
	for(auto tmpPtr = statistics.begin(); tmpPtr != statistics.end();)
	{
		if(tmpPtr->second < PARALLAX_THRESHOLD)
			tmpPtr = statistics.erase(tmpPtr);	
		else 
			++tmpPtr;
	}
	for(auto ptr = edgePoint.end() - 1; ;)		//对最后一个flagPoint之后的像素点重新做统计，剔除视差部分
	{
		if(statistics.find(ptr->y) == statistics.end())
			ptr = edgePoint.erase(ptr) - 1;
		else
			--ptr;
		if(*ptr == flagPoint[flagPoint.size() - 1])
		{
			edgePoint.erase(ptr);
			break;
		}
	}
	for(auto ptr = edgePoint.end() - 1; ptr != edgePoint.begin(); --ptr)
	{
		if(ptr->x < flagPoint[flagPoint.size() - 1].x)
		{
			flagPoint[flagPoint.size() - 1] = *(ptr + 1);
			break;
		}
	}
	cout << "internal_measure running duration is " << static_cast<double>(clock() - beg)/CLOCKS_PER_SEC << " sec" << endl;
	//对flagPoint进行一次检测
	if(flagPoint.size() != 4)	//flagPoint数不等于4，一般是canny参数设置的不合理 或者 有其他不明物体侵入检测视野导致的
		cout << "flagPoint error!" << endl;
	else
		cout << "Cylinder's diameter is:" << flagPoint[2].x - flagPoint[1].x << endl;	//得到圆柱直径
	//圆柱直径要考虑视差近大远小导致的误差，可能需要校正

	
	Mat showImage;
	cvtColor(cannyImage, showImage, CV_GRAY2BGR);
	for(auto i = flagPoint.cbegin(); i < flagPoint.cend(); ++i)
	{
		circle(showImage, Point(i->y, i->x), 1, Scalar(0, 0, 255), -1);
		cout << *i;
	}
	namedWindow("canny", WINDOW_NORMAL);
	imshow("canny", showImage);
	
	cout << endl;
}
/*
 *零件边界要清晰，以canny可以正常分辨为准
 *以下程序依赖于假设：canny检测的零件顶面外轮廓是基本封闭的 以及 零件图被正确校正到垂直位置
 *形变检测上下以及左右对称程度。不对称达到一定程度判定为形变
 *零件外接矩形测量的是零件最长的两边的长度
 */
//计算零件尺寸的实际值的时候，要考虑零件高度导致的大小变化。
#define DEFORMATION_THRESHOLD 3		//形变阈值
#define SIZE_POINT_NUM        15	//参与判断最外的点是否属于零件轮廓的点数量 15
#define DIFF_THRESH           0.02	//SIZE_POINT_NUM个点，其中最大值的点的方差小于这个值，就认为该最大值点属于零件轮廓 0.02
void top_measure(Mat &srcImage, bool sizeFlag, bool deformationFlag, int *sidePos, double angle)
{
	clock_t beg = clock();
	Mat cannyImage, showImage;
	Canny(srcImage, cannyImage, 110, 110, 3, true);
	angleCalibration(cannyImage, angle);	
	cvtColor(cannyImage, showImage, CV_GRAY2BGR);
	
	Point2i vertices[4]; 
	vertices[0] = Point2i(0, cannyImage.rows - 1);
	vertices[1] = Point2i(0, 0);
	vertices[2] = Point2i(cannyImage.cols - 1, 0);
	vertices[3] = Point2i(cannyImage.rows - 1, cannyImage.cols - 1);
	vector<Point> leftLocation;
	leftLocation.reserve(1024);
	vector<Point> rightLocation;
	rightLocation.reserve(1024);
	vector<Point> upLocation;	
	upLocation.reserve(1024);
	vector<Point> downLocation;	
	downLocation.reserve(1024);
	//这里不一定需要每个点都扫，为了提高效率可以隔2/3/4个点扫一次
	for(auto rowCounter = vertices[2].y; rowCounter <= vertices[0].y; ++rowCounter)	//扫描外接矩形内部边缘点（横向扫描,探测竖直方向形变量）
	{
		for(auto colCounterLeft = vertices[0].x + 10; colCounterLeft <= vertices[2].x - 10; ++colCounterLeft)	//从左往右
		{
			if(cannyImage.at<unsigned char>(rowCounter, colCounterLeft))
			{
				for(auto colCounterRight = vertices[2].x; colCounterRight >= vertices[0].x; --colCounterRight)	//从右往左
				{
					if(cannyImage.at<unsigned char>(rowCounter, colCounterRight)) 
					{
						leftLocation.push_back(Point(colCounterLeft, rowCounter));
						rightLocation.push_back(Point(colCounterRight, rowCounter));
						break;
					}
				}
				break;
			}
		}
	}

	for(auto colCounter = vertices[0].x; colCounter <= vertices[2].x; ++colCounter)	//扫描外接矩形内部边缘点（竖直扫描,探测竖直方向形变量）
	{
		for(auto rowCounterUp = vertices[1].y + 10; rowCounterUp <= vertices[0].y - 10; ++rowCounterUp)	//从上往下
		{
			if(cannyImage.at<unsigned char>(rowCounterUp, colCounter))
			{
				for(auto rowCounterDown = vertices[0].y; rowCounterDown >= vertices[1].y; --rowCounterDown)	//从下往上
				{
					if(cannyImage.at<unsigned char>(rowCounterDown, colCounter))
					{
						upLocation.push_back(Point(colCounter, rowCounterUp));
						downLocation.push_back(Point(colCounter, rowCounterDown));
						break;
					}
				}
				break;
			}
		}
	}
	//下面根据扫得的边缘点坐标，求零件外接矩形尺寸
	while(1)
	{
		partial_sort(downLocation.begin(), downLocation.begin() + SIZE_POINT_NUM, downLocation.end(), [](Point a, Point b){return a.y > b.y;});
		double means = accumulate(downLocation.begin(), downLocation.begin() + SIZE_POINT_NUM, 0.0, [](int init, Point a){return init + a.y;}) / SIZE_POINT_NUM;
		if(abs(downLocation.begin()->y - means) / SIZE_POINT_NUM > DIFF_THRESH) //确保最大值点不是离散在零件轮廓外的点
		{
			downLocation.erase(downLocation.begin());
		}
		else
			break;
	}
	int downSide = downLocation.begin()->y;
	line(showImage, Point(0, downSide), Point(cannyImage.cols, downSide), Scalar(0, 255, 255));
	while(1)
	{
		partial_sort(rightLocation.begin(), rightLocation.begin() + SIZE_POINT_NUM, rightLocation.end(), [](Point a, Point b){return a.x > b.x;});
		double means = accumulate(rightLocation.begin(), rightLocation.begin() + SIZE_POINT_NUM, 0.0, [](int init, Point a){return init + a.x;}) / SIZE_POINT_NUM;
		if(abs(rightLocation.begin()->x - means) / SIZE_POINT_NUM > DIFF_THRESH) //确保最大值点不是离散在零件轮廓外的点
		{
			rightLocation.erase(rightLocation.begin());
		}
		else
			break;
	}
	int rightSide = rightLocation.begin()->x;
	line(showImage, Point(rightSide, 0), Point(rightSide, cannyImage.rows), Scalar(0, 255, 255));
	while(1)
	{
		partial_sort(upLocation.begin(), upLocation.begin() + SIZE_POINT_NUM, upLocation.end(), [](Point a, Point b){return a.y < b.y;});
		double means = accumulate(upLocation.begin(), upLocation.begin() + SIZE_POINT_NUM, 0.0, [](int init, Point a){return init + a.y;}) / SIZE_POINT_NUM;
		if(abs(upLocation.begin()->y - means) / SIZE_POINT_NUM > DIFF_THRESH) //确保最大值点不是离散在零件轮廓外的点
		{
			upLocation.erase(upLocation.begin());
		}
		else
			break;
	}
	int upSide = upLocation.begin()->y;
	line(showImage, Point(0, upSide), Point(cannyImage.cols, upSide), Scalar(0, 255, 255));
	while(1)
	{
		partial_sort(leftLocation.begin(), leftLocation.begin() + SIZE_POINT_NUM, leftLocation.end(),  [](Point a, Point b){return a.x < b.x;});
		double means = accumulate(leftLocation.begin(), leftLocation.begin() + SIZE_POINT_NUM, 0.0, [](int init, Point a){return init + a.x;}) / SIZE_POINT_NUM;
		if(abs(leftLocation.begin()->x - means) / SIZE_POINT_NUM > DIFF_THRESH) //确保最大值点不是离散在零件轮廓外的点
		{
			leftLocation.erase(leftLocation.begin());
		}
		else
			break;
	}
	int leftSide = leftLocation.begin()->x;
	line(showImage, Point(leftSide, 0), Point(leftSide, cannyImage.rows), Scalar(0, 255, 255));
	cout << "The size is [" << rightSide - leftSide << "x" << downSide - upSide << "]" << endl;
	
	if(sidePos != NULL)
	{
		sidePos[0] = leftSide;
		sidePos[1] = rightSide;
		sidePos[2] = upSide;
		sidePos[3] = downSide;
	}

	if(deformationFlag)
	{
		int deformation = 0;
		vector<int> deformationVertical;
		sort(downLocation.begin(), downLocation.end(), [](Point a, Point b){return a.x < b.x;});
		auto tempDownLeft = downLocation.begin();
		auto tempDownRight = downLocation.rbegin();
		for(; tempDownLeft != downLocation.end() && tempDownRight != downLocation.rend();) 
		{
			if(tempDownLeft->x < leftSide || tempDownLeft->x > rightSide || tempDownLeft->y < upSide || tempDownLeft->y > downSide)
			{
				++tempDownLeft;
				continue;
			}	
			if(tempDownRight->x > rightSide || tempDownRight->x < leftSide || tempDownRight->y < upSide || tempDownRight->y > downSide) 
			{
				++tempDownRight;
				continue;
			}
			if(abs(tempDownLeft->y - tempDownRight->y) > DEFORMATION_THRESHOLD)
			{
				//cout << "Up def detected: " << tempDownLeft->y - tempDownRight->y << endl;
				++deformation;
				circle(showImage, *tempDownLeft, 1, Scalar(0, 255, 0), -1);
			}
			else if(deformation)
			{
				deformationVertical.push_back(deformation);
				deformation = 0;
			}
			++tempDownLeft;
			++tempDownRight;
		}
		if(deformation)
		{
			deformationVertical.push_back(deformation);
			deformation = 0;
		}

		vector<int> deformationHorizon;
		sort(upLocation.begin(), upLocation.end(), [](Point a, Point b){return a.x < b.x;});
		auto tempUpLeft = upLocation.begin();
		auto tempUpRight = upLocation.rbegin();
		for(; tempUpLeft != upLocation.end() && tempUpRight != upLocation.rend(); )
		{
			if(tempUpLeft->y < upSide || tempUpLeft->y > downSide || tempUpLeft->x < leftSide || tempUpLeft->x > rightSide)
			{
				++tempUpLeft;	//Down指针不动，up指针跳过一个
				continue;
			}
			if(tempUpRight->y > downSide || tempUpRight->y < upSide || tempUpRight->x < leftSide || tempUpRight->x > rightSide)
			{
				++tempUpRight;
				continue;
			}
			if(abs(tempUpLeft->y - tempUpRight->y) > DEFORMATION_THRESHOLD)
			{
				//cout << "Down def detected: " << tempUpLeft->y - tempUpRight->y << endl;
				++deformation;
				circle(showImage, *tempUpLeft, 1, Scalar(0, 0, 255), -1);
			}
			else if(deformation)
			{
				deformationHorizon.push_back(deformation);
				deformation = 0;
			}
			++tempUpLeft;
			++tempUpRight;
		}
		if(deformation)
		{
			deformationHorizon.push_back(deformation);
			deformation = 0;
		}
	}

	cout << "The top_measure function running duration is "  << static_cast<double>(clock() - beg)/CLOCKS_PER_SEC << " sec" <<endl;

	namedWindow("main_win", WINDOW_NORMAL);
	imshow("main_win", showImage);
	//namedWindow("canny", WINDOW_NORMAL);
	//imshow("canny", cannyImage);
}
