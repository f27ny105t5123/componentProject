#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "crackDetect.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void RemoveSmallRegion(Mat& Src, Mat& Dst, int AreaLimit, int CheckMode, int NeihborMode)
{
	int RemoveCount = 0;       //记录除去的个数  
	//记录每个像素点检验状态的标签，0代表未检查，1代表正在检查,2代表检查不合格（需要反转颜色），3代表检查合格或不需检查  
	Mat Pointlabel = Mat::zeros(Src.size(), CV_8UC1);

	if (CheckMode == 1)
	{
		//cout << "Mode: 去除小区域. ";
		for (int i = 0; i < Src.rows; ++i)
		{
			uchar* iData = Src.ptr<uchar>(i);
			uchar* iLabel = Pointlabel.ptr<uchar>(i);
			for (int j = 0; j < Src.cols; ++j)
				if (iData[j] < 10)
					iLabel[j] = 3;
		}
	}
	else
	{
		//cout << "Mode: 去除孔洞. ";
		for (int i = 0; i < Src.rows; ++i)
		{
			uchar* iData = Src.ptr<uchar>(i);
			uchar* iLabel = Pointlabel.ptr<uchar>(i);
			for (int j = 0; j < Src.cols; ++j)
			{
				if (iData[j] > 10)
				{
					iLabel[j] = 3;
				}
			}
		}
	}

	//vector<Point> NeihborPos;  //记录邻域点位置  
	//NeihborPos.push_back(Point(-1, 0));
	//NeihborPos.push_back(Point(1, 0));
	//NeihborPos.push_back(Point(0, -1));
	//NeihborPos.push_back(Point(0, 1));
	//if (NeihborMode == 1)
	//{
	//	/*	cout << "Neighbor mode: 8邻域." << endl;*/
	//	NeihborPos.push_back(Point2i(-1, -1));
	//	NeihborPos.push_back(Point2i(-1, 1));
	//	NeihborPos.push_back(Point2i(1, -1));
	//	NeihborPos.push_back(Point2i(1, 1));
	//}
	//else cout << "Neighbor mode: 4邻域." << endl;
	int NeihborPos[8][2] = {-1, 0, 1, 0, 0, -1, 0, 1, -1, -1, -1, 1, 1, -1, 1, 1};  //记录邻域点位置
	int ydiff =  Pointlabel.ptr<uchar>(1) - Pointlabel.ptr<uchar>(0);
	int diff[8];
	for (int i = 0; i < 8; i++) {
		diff[i] = NeihborPos[i][0] + NeihborPos[i][1] * ydiff;
	}

	int NeihborCount = 4 + 4 * NeihborMode;
	int CurrX = 0, CurrY = 0;
	//开始检测  
	for (int i = 0; i < Src.rows; ++i)
	{
		uchar* iLabel = Pointlabel.ptr<uchar>(i);
		for (int j = 0; j < Src.cols; ++j)
		{
			if (iLabel[j] == 0)
			{
				//********开始该点处的检查**********  
				//vector<Point2i> GrowBuffer;                                      //堆栈，用于存储生长点  
				//GrowBuffer.push_back(Point2i(j, i));
				int GrowBuffer[8192][2];
				int num = 0;
				iLabel[j] = 1;
				int CheckResult = 0;                                               //用于判断结果（是否超出大小），0为未超出，1为超出  

				GrowBuffer[num][0] = j;
				GrowBuffer[num++][1] = i;
				for (int z = 0; z<num; z++)
				{
					uchar* iLabel2 = Pointlabel.ptr<uchar>(GrowBuffer[z][1]) + GrowBuffer[z][0];
					for (int q = 0; q<NeihborCount; q++)                                      //检查四个邻域点  
					{
						CurrX = GrowBuffer[z][0] + NeihborPos[q][0];
						CurrY = GrowBuffer[z][1] + NeihborPos[q][1];
						if (CurrX >= 0 && CurrX<Src.cols&&CurrY >= 0 && CurrY<Src.rows)  //防止越界  
						{
							if (iLabel2[diff[q]] == 0)
							{
								//GrowBuffer.push_back(Point2i(CurrX, CurrY));  //邻域点加入buffer  
								GrowBuffer[num][0] = CurrX;
								GrowBuffer[num++][1] = CurrY;
								iLabel2[diff[q]] = 1;           //更新邻域点的检查标签，避免重复检查  
							}
						}
					}

				}
				if (num>AreaLimit) CheckResult = 2;                 //判断结果（是否超出限定的大小），1为未超出，2为超出  
				else { CheckResult = 1;   RemoveCount++; }
				for (int z = 0; z<num; z++)                         //更新Label记录  
				{
					Pointlabel.at<uchar>(GrowBuffer[z][1], GrowBuffer[z][0]) += CheckResult;
				}
				//********结束该点处的检查**********  


			}
		}
	}

	CheckMode = 255 * (1 - CheckMode);
	//开始反转面积过小的区域  
	for (int i = 0; i < Src.rows; ++i)
	{
		uchar* iData = Src.ptr<uchar>(i);
		uchar* iDstData = Dst.ptr<uchar>(i);
		uchar* iLabel = Pointlabel.ptr<uchar>(i);
		for (int j = 0; j < Src.cols; ++j)
		{
			if (iLabel[j] == 2)
			{
				iDstData[j] = CheckMode;
			}
			else if (iLabel[j] == 3)
			{
				iDstData[j] = iData[j];
			}
		}
	}

	/*cout << RemoveCount << " objects removed." << endl;*/
}
void RemoveSmallRegion1(Mat& Src, Mat& Dst, int AreaLimit, int CheckMode, int NeihborMode)  //ChenkMode0代表去除黑区域，1代表去除白色区域
{
	int RemoveCount = 0;       //记录除去的个数  
	//记录每个像素点检验状态的标签，0代表未检查，1代表正在检查,2代表检查不合格（需要反转颜色），3代表检查合格或不需检查  
	Mat Pointlabel = Mat::zeros(Src.size(), CV_8UC1);

	if (CheckMode == 1)
	{
		//cout << "Mode: 去除小区域. ";
		for (int i = 0; i < Src.rows; ++i)
		{
			uchar* iData = Src.ptr<uchar>(i);
			uchar* iLabel = Pointlabel.ptr<uchar>(i);
			for (int j = 0; j < Src.cols; ++j)
			if (iData[j] < 10)
				iLabel[j] = 3;
		}
	}
	else
	{
		//cout << "Mode: 去除孔洞. ";
		for (int i = 0; i < Src.rows; ++i)
		{
			uchar* iData = Src.ptr<uchar>(i);
			uchar* iLabel = Pointlabel.ptr<uchar>(i);
			for (int j = 0; j < Src.cols; ++j)
			{
				if (iData[j] > 10)
				{
					iLabel[j] = 3;
				}
			}
		}
	}

	vector<Point> NeihborPos;  //记录邻域点位置  
	NeihborPos.push_back(Point(-1, 0));
	NeihborPos.push_back(Point(1, 0));
	NeihborPos.push_back(Point(0, -1));
	NeihborPos.push_back(Point(0, 1));
	if (NeihborMode == 1)
	{
		//cout << "Neighbor mode: 8邻域." << endl;
		NeihborPos.push_back(Point2i(-1, -1));
		NeihborPos.push_back(Point2i(-1, 1));
		NeihborPos.push_back(Point2i(1, -1));
		NeihborPos.push_back(Point2i(1, 1));
	}
	else cout << "Neighbor mode: 4邻域." << endl;
	int NeihborCount = 4 + 4 * NeihborMode;
	int CurrX = 0, CurrY = 0;
	//开始检测  
	for (int i = 0; i < Src.rows; ++i)
	{
		uchar* iLabel = Pointlabel.ptr<uchar>(i);
		for (int j = 0; j < Src.cols; ++j)
		{
			if (iLabel[j] == 0)
			{
				//********开始该点处的检查**********  
				vector<Point2i> GrowBuffer;                                      //堆栈，用于存储生长点  
				GrowBuffer.push_back(Point2i(j, i));
				Pointlabel.at<uchar>(i, j) = 1;
				int CheckResult = 0;                                               //用于判断结果（是否超出大小），0为未超出，1为超出  

				for (int z = 0; z<GrowBuffer.size(); z++)
				{

					for (int q = 0; q<NeihborCount; q++)                                      //检查四个邻域点  
					{
						CurrX = GrowBuffer.at(z).x + NeihborPos.at(q).x;
						CurrY = GrowBuffer.at(z).y + NeihborPos.at(q).y;
						if (CurrX >= 0 && CurrX<Src.cols&&CurrY >= 0 && CurrY<Src.rows)  //防止越界  
						{
							if (Pointlabel.at<uchar>(CurrY, CurrX) == 0)
							{
								GrowBuffer.push_back(Point2i(CurrX, CurrY));  //邻域点加入buffer  
								Pointlabel.at<uchar>(CurrY, CurrX) = 1;           //更新邻域点的检查标签，避免重复检查  
							}
						}
					}

				}
				if (GrowBuffer.size()>AreaLimit) CheckResult = 2;                 //判断结果（是否超出限定的大小），1为未超出，2为超出  
				else { CheckResult = 1;   RemoveCount++; }
				for (int z = 0; z<GrowBuffer.size(); z++)                         //更新Label记录  
				{
					CurrX = GrowBuffer.at(z).x;
					CurrY = GrowBuffer.at(z).y;
					Pointlabel.at<uchar>(CurrY, CurrX) += CheckResult;
				}
				//********结束该点处的检查**********  


			}
		}
	}

	CheckMode = 255 * (1 - CheckMode);
	//开始反转面积过小的区域  
	for (int i = 0; i < Src.rows; ++i)
	{
		uchar* iData = Src.ptr<uchar>(i);
		uchar* iDstData = Dst.ptr<uchar>(i);
		uchar* iLabel = Pointlabel.ptr<uchar>(i);
		for (int j = 0; j < Src.cols; ++j)
		{
			if (iLabel[j] == 2)
			{
				iDstData[j] = CheckMode;
			}
			else if (iLabel[j] == 3)
			{
				iDstData[j] = iData[j];
			}
		}
	}

	//cout << RemoveCount << " objects removed." << endl;
}
int maxval(int *ptr, int num)
{
	int max = ptr[0];
	for (int i = 1; i < num; i++)
	if (max < ptr[i])
		max = ptr[i];
	return max;
}
Mat histcount(Mat &img, int *ptri, uchar *ptr, int &count)   //画灰度图的直方图,ptri为存放图像灰度级与其数量的对应关系的数组名，ptr指向图像第一个像素。
{
	int maxheight = 256;
	for (int i = 0; i < img.rows*img.cols; i++)
	{
		if (ptr[i] != 255)
		{
			++ptri[ptr[i]];
			count++;
		}
	}	                                 //统计原图各灰度级像素点个数
	ptri[255] = 0;
	int max = maxval(ptri, 256);
	Mat hist(256, 512, CV_8UC3);                         //直方图宽512，高256
	for (int i = 0; i < 256; i++)
	{
		double height = ptri[i] * maxheight / (1.0*max);
		rectangle(hist, Point(i * 2, 255), Point((i + 1) * 2 - 1, 255 - height), Scalar(0, 0, 255));
	}
	return hist;
}
void histogramequ(Mat &img, Mat img_htg)                       //直方图均衡函数，输入输出图像大小一致
{
	int count = 0;
	uchar* ptr = img.ptr<uchar>(0);
	uchar* ptr_htg = img_htg.ptr<uchar>(0);                   //用以存放原图各灰度级像素点个数
	int grayori[256] = { 0 };
	double  grayp[256] = { 0 };
	int graynew[256] = { 0 };                                //用以直方图均衡后各灰度级映射关系
	Mat histori/*(256, 512, CV_8UC3)*/;
	histori = histcount(img, grayori, ptr, count);
	//	imshow("hist", histori);                                //画原图像直方图
	for (int i = 0; i < 256; i++)
		grayp[i] = grayori[i] / (1.0*count);               //原图各灰度级像素点个数占总像素个数比率
	double sum = 0;
	int sumi;
	for (int i = 0; i < 256; i++)
	{
		sum += grayp[i];
		sumi = int(255 * sum + 0.5);

		graynew[i] = sumi;                      //构造映射关系
	}
	Mat hist_htg/*(256, 512, CV_8UC3)*/;
	for (int i = 0; i < img.rows*img.cols; i++)        //直方图均衡
	{
		if (ptr_htg[i] != 255)
		{
			ptr_htg[i] = graynew[ptr[i]];
		}
	}
	count = 0;
	memset(graynew, 0, sizeof(int)* 256);               //将存储映射关系的数组清零用以存放均衡后图像的直方图统计数据
	hist_htg = histcount(img_htg, graynew, ptr_htg, count);
	//	imshow("hist_htg", hist_htg);                     //均衡后图像直方图
}
Mat BackgroundEqu(Mat &srcImage)                                  //背景光均衡
{
	const int subWindowSize = 8;	//子窗口大小
	//子窗口灰度均值
	Mat subMaskIntensity(srcImage.rows / subWindowSize, srcImage.cols / subWindowSize, CV_8UC1);
	for (auto i = 0; i < subMaskIntensity.rows; ++i)
	{
		uchar *ptrSubmask = subMaskIntensity.ptr<uchar>(i);
		for (auto j = 0; j < subMaskIntensity.cols; ++j)
		{
			long long intensity = 0;
			for (auto k = 0; k < subWindowSize; ++k)
			{
				uchar *ptrSrc = srcImage.ptr<uchar>(i*subWindowSize + k);
				for (auto m = 0; m < subWindowSize; ++m)
				{
					intensity += *(ptrSrc+j*subWindowSize + m);
				}
			}
			*(ptrSubmask+j) = intensity / (subWindowSize * subWindowSize);
		}
	}

	for (auto i = 1; i < subMaskIntensity.rows - 1; ++i)
	{
		uchar *ptrSubmask = subMaskIntensity.ptr<uchar>(i);
		for (auto j = 1; j < subMaskIntensity.cols - 1; ++j)
		{
			if ((*(ptrSubmask + j)< *(ptrSubmask + j + 1)) && (*(ptrSubmask + j)< *(ptrSubmask + j - 1)))
			{
				*(ptrSubmask + j) = (*(ptrSubmask + j + 1) + *(ptrSubmask + j - 1)) / 2;
			}
			if ((*(ptrSubmask + j)<*(ptrSubmask + j - subMaskIntensity.cols)) && (*(ptrSubmask + j)<*(ptrSubmask + j + subMaskIntensity.cols)))
			{
				*(ptrSubmask + j) = (*(ptrSubmask + j - subMaskIntensity.cols) + *(ptrSubmask + j + subMaskIntensity.cols)) / 2;
			}
		}
	}

	long long imgIntensity = 0;			//全图灰度均值
	for (auto i = 0; i < srcImage.rows*srcImage.cols; ++i)
	{
		uchar *ptrSrc = srcImage.ptr<uchar>(0);
		imgIntensity += *(ptrSrc+i);
	}
	imgIntensity /= (srcImage.cols * srcImage.rows);

	for (auto i = 0; i < subMaskIntensity.rows; ++i)		//消除背景光
	{
		uchar *ptrSubmask = subMaskIntensity.ptr<uchar>(i);
		for (auto j = 0; j < subMaskIntensity.cols; ++j)
		{
			double balanceFactor = double(imgIntensity) / *(ptrSubmask+j);
			for (auto k = 0; k < subWindowSize; ++k)
			{
				uchar *ptrSrc = srcImage.ptr<uchar>(i*subWindowSize + k);
				for (auto m = 0; m < subWindowSize; ++m)
				{
					*(ptrSrc+j*subWindowSize+m) *= balanceFactor;

				}
			}
		}
	}
	return srcImage;
}
Mat SilTest(Mat &img)
{
	int64 timestart = 0, timeend = 0;
	double t1, t2, t3;
	int grayThreshold = 45;             //25
	int areaLimit = 6;                 //12
	Mat img_copy,img_1;
	img.copyTo(img_copy);
	//imshow("backgroundequ", img_copy);
	timestart = getTickCount();
	histogramequ(img_copy, img_copy);
	timeend = getTickCount();
	t1 = 1000.0*(timeend - timestart) / getTickFrequency();
	//imshow("htg", img_copy);

	threshold(img_copy, img_copy,grayThreshold, 255, THRESH_BINARY);                              //15            32
	//imshow("bin", img_copy);
	RemoveSmallRegion1(img_copy, img_copy, areaLimit, 0, 0);                       //15                         10   
	timestart = timeend;
	timeend = getTickCount();
	t2 = 1000.0*(timeend - timestart) / getTickFrequency();
	//imshow("remove", img_copy);
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
	//morphologyEx(img_copy, img_copy, MORPH_CLOSE, element);
	erode(img_copy, img_1, element);                                          //3
	RemoveSmallRegion(img_1, img_1, 180, 0, 1);                                  //100                   160
	timestart = timeend;
	timeend = getTickCount();
	t3 = 1000.0*(timeend - timestart) / getTickFrequency();
	cout << "[Sil]t1: " << t1 << " ms, t2: " << t2 << " ms, t3: " << t3 << endl;
	imshow("result", img_1);
	return img_1;
}
void ISilt(Mat &srcImage)
{
	vector<int> statistic;			//垂直窗口统计
	vector<int> HorizonSta;
 	const int staWindowSize = 10;
	const int WindowPixels = staWindowSize*srcImage.cols;
	uchar *ptr = srcImage.ptr<uchar>(0);
	int counter = 0;
	for (auto i = 0; i < srcImage.rows * srcImage.cols; ++i)
	{	
		if (*(ptr+i) < 20)
			counter++;
		if ((i+1)%WindowPixels==0)
		{
			statistic.push_back(counter);
//			cout << counter << ",";		//输出统计数据
			counter = 0;
		}
	}
	cout << endl;
	for (auto i = 1; i < statistic.size(); ++i)
	{
		float ForwardRatio = statistic[i] * 1.0 / (statistic[i - 1]+1);
		if (ForwardRatio > 4)		//判断裂缝的阈值
		{
			auto j = i;
			float BackwardRatio;
			do
			{
				BackwardRatio = statistic[j] * 1.0 / (statistic[j + 1] + 1);
				++j;
			} while (BackwardRatio < 4);
			//for (; j < statistic.size(); ++j)
			//{
			//	if (statistic[j] < 155) break;		//裂缝结束阈值
			//}
			if ((j - i > 3) && (j - i < 8))          //进一步横向窗口判断
			{
				ptr = srcImage.ptr<uchar>(i*10);
				counter = 0;
				const int RowSize = (j - i) * 10;
				for (size_t n = 0; n < srcImage.cols / 10; n++)
				{
					for (size_t m = 0; m < RowSize; m++)
					{
						ptr = srcImage.ptr<uchar>(i * 10 + m);
						for (size_t k = 0; k < 10; k++)
						{
							if (*(ptr + n * 10 + k)<20)
							counter ++ ;
						}
					}
					HorizonSta.push_back(counter);
					cout << counter<<"  ";
					counter = 0;
				}
				int num = 0;
				for (size_t i = 0; i < HorizonSta.size(); i++)
				{
					if (HorizonSta[i]>100)
						num++;
				}
				float Isilt = num*1.0 / HorizonSta.size();
				if (Isilt > 1 / 3)
				{
					cout << endl << "裂缝在" << i * 10 << "行到" << j * 10 << "行之间" << endl;
					cvtColor(srcImage, srcImage, CV_GRAY2BGR);
					line(srcImage, Point(0, j * 10), Point(srcImage.cols, j * 10), Scalar(0, 0, 255), 2, 4);
					line(srcImage, Point(0, i * 10), Point(srcImage.cols, i * 10), Scalar(0, 0, 255), 2, 4);
					break;
				}

			}
		}
	}
	imshow("result", srcImage);
	
}
void detector(Mat &img)
{
	int timestart = 0, timeend = 0;
	double t1, t2, t3;
	//string imgname = "D:\\G disc\\工业视觉检测\\零件11.13\\截\\19.1.bmp";
	//if (argc > 1)
	//	imgname = argv[1];
	//Mat img = imread(imgname, 0);
	//imshow("ori", img);
	timestart = getTickCount();
	img = BackgroundEqu(img);
	timeend = getTickCount();
	t1 = 1000.0*(timeend - timestart) / getTickFrequency();
	img = SilTest(img);
	timestart = timeend;
	timeend = getTickCount();
	t2 = 1000.0*(timeend - timestart) / getTickFrequency();
	ISilt(img);
	timestart = timeend;
	timeend = getTickCount();
	t3 = 1000.0*(timeend - timestart) / getTickFrequency();
	cout << "\nt1: " << t1 << " ms, t2: " << t2 << " ms, t3: " << t3 << endl;
	waitKey(0);
}
