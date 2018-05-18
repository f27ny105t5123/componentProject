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
	int RemoveCount = 0;       //��¼��ȥ�ĸ���  
	//��¼ÿ�����ص����״̬�ı�ǩ��0����δ��飬1�������ڼ��,2�����鲻�ϸ���Ҫ��ת��ɫ����3������ϸ������  
	Mat Pointlabel = Mat::zeros(Src.size(), CV_8UC1);

	if (CheckMode == 1)
	{
		//cout << "Mode: ȥ��С����. ";
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
		//cout << "Mode: ȥ���׶�. ";
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

	//vector<Point> NeihborPos;  //��¼�����λ��  
	//NeihborPos.push_back(Point(-1, 0));
	//NeihborPos.push_back(Point(1, 0));
	//NeihborPos.push_back(Point(0, -1));
	//NeihborPos.push_back(Point(0, 1));
	//if (NeihborMode == 1)
	//{
	//	/*	cout << "Neighbor mode: 8����." << endl;*/
	//	NeihborPos.push_back(Point2i(-1, -1));
	//	NeihborPos.push_back(Point2i(-1, 1));
	//	NeihborPos.push_back(Point2i(1, -1));
	//	NeihborPos.push_back(Point2i(1, 1));
	//}
	//else cout << "Neighbor mode: 4����." << endl;
	int NeihborPos[8][2] = {-1, 0, 1, 0, 0, -1, 0, 1, -1, -1, -1, 1, 1, -1, 1, 1};  //��¼�����λ��
	int ydiff =  Pointlabel.ptr<uchar>(1) - Pointlabel.ptr<uchar>(0);
	int diff[8];
	for (int i = 0; i < 8; i++) {
		diff[i] = NeihborPos[i][0] + NeihborPos[i][1] * ydiff;
	}

	int NeihborCount = 4 + 4 * NeihborMode;
	int CurrX = 0, CurrY = 0;
	//��ʼ���  
	for (int i = 0; i < Src.rows; ++i)
	{
		uchar* iLabel = Pointlabel.ptr<uchar>(i);
		for (int j = 0; j < Src.cols; ++j)
		{
			if (iLabel[j] == 0)
			{
				//********��ʼ�õ㴦�ļ��**********  
				//vector<Point2i> GrowBuffer;                                      //��ջ�����ڴ洢������  
				//GrowBuffer.push_back(Point2i(j, i));
				int GrowBuffer[8192][2];
				int num = 0;
				iLabel[j] = 1;
				int CheckResult = 0;                                               //�����жϽ�����Ƿ񳬳���С����0Ϊδ������1Ϊ����  

				GrowBuffer[num][0] = j;
				GrowBuffer[num++][1] = i;
				for (int z = 0; z<num; z++)
				{
					uchar* iLabel2 = Pointlabel.ptr<uchar>(GrowBuffer[z][1]) + GrowBuffer[z][0];
					for (int q = 0; q<NeihborCount; q++)                                      //����ĸ������  
					{
						CurrX = GrowBuffer[z][0] + NeihborPos[q][0];
						CurrY = GrowBuffer[z][1] + NeihborPos[q][1];
						if (CurrX >= 0 && CurrX<Src.cols&&CurrY >= 0 && CurrY<Src.rows)  //��ֹԽ��  
						{
							if (iLabel2[diff[q]] == 0)
							{
								//GrowBuffer.push_back(Point2i(CurrX, CurrY));  //��������buffer  
								GrowBuffer[num][0] = CurrX;
								GrowBuffer[num++][1] = CurrY;
								iLabel2[diff[q]] = 1;           //���������ļ���ǩ�������ظ����  
							}
						}
					}

				}
				if (num>AreaLimit) CheckResult = 2;                 //�жϽ�����Ƿ񳬳��޶��Ĵ�С����1Ϊδ������2Ϊ����  
				else { CheckResult = 1;   RemoveCount++; }
				for (int z = 0; z<num; z++)                         //����Label��¼  
				{
					Pointlabel.at<uchar>(GrowBuffer[z][1], GrowBuffer[z][0]) += CheckResult;
				}
				//********�����õ㴦�ļ��**********  


			}
		}
	}

	CheckMode = 255 * (1 - CheckMode);
	//��ʼ��ת�����С������  
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
void RemoveSmallRegion1(Mat& Src, Mat& Dst, int AreaLimit, int CheckMode, int NeihborMode)  //ChenkMode0����ȥ��������1����ȥ����ɫ����
{
	int RemoveCount = 0;       //��¼��ȥ�ĸ���  
	//��¼ÿ�����ص����״̬�ı�ǩ��0����δ��飬1�������ڼ��,2�����鲻�ϸ���Ҫ��ת��ɫ����3������ϸ������  
	Mat Pointlabel = Mat::zeros(Src.size(), CV_8UC1);

	if (CheckMode == 1)
	{
		//cout << "Mode: ȥ��С����. ";
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
		//cout << "Mode: ȥ���׶�. ";
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

	vector<Point> NeihborPos;  //��¼�����λ��  
	NeihborPos.push_back(Point(-1, 0));
	NeihborPos.push_back(Point(1, 0));
	NeihborPos.push_back(Point(0, -1));
	NeihborPos.push_back(Point(0, 1));
	if (NeihborMode == 1)
	{
		//cout << "Neighbor mode: 8����." << endl;
		NeihborPos.push_back(Point2i(-1, -1));
		NeihborPos.push_back(Point2i(-1, 1));
		NeihborPos.push_back(Point2i(1, -1));
		NeihborPos.push_back(Point2i(1, 1));
	}
	else cout << "Neighbor mode: 4����." << endl;
	int NeihborCount = 4 + 4 * NeihborMode;
	int CurrX = 0, CurrY = 0;
	//��ʼ���  
	for (int i = 0; i < Src.rows; ++i)
	{
		uchar* iLabel = Pointlabel.ptr<uchar>(i);
		for (int j = 0; j < Src.cols; ++j)
		{
			if (iLabel[j] == 0)
			{
				//********��ʼ�õ㴦�ļ��**********  
				vector<Point2i> GrowBuffer;                                      //��ջ�����ڴ洢������  
				GrowBuffer.push_back(Point2i(j, i));
				Pointlabel.at<uchar>(i, j) = 1;
				int CheckResult = 0;                                               //�����жϽ�����Ƿ񳬳���С����0Ϊδ������1Ϊ����  

				for (int z = 0; z<GrowBuffer.size(); z++)
				{

					for (int q = 0; q<NeihborCount; q++)                                      //����ĸ������  
					{
						CurrX = GrowBuffer.at(z).x + NeihborPos.at(q).x;
						CurrY = GrowBuffer.at(z).y + NeihborPos.at(q).y;
						if (CurrX >= 0 && CurrX<Src.cols&&CurrY >= 0 && CurrY<Src.rows)  //��ֹԽ��  
						{
							if (Pointlabel.at<uchar>(CurrY, CurrX) == 0)
							{
								GrowBuffer.push_back(Point2i(CurrX, CurrY));  //��������buffer  
								Pointlabel.at<uchar>(CurrY, CurrX) = 1;           //���������ļ���ǩ�������ظ����  
							}
						}
					}

				}
				if (GrowBuffer.size()>AreaLimit) CheckResult = 2;                 //�жϽ�����Ƿ񳬳��޶��Ĵ�С����1Ϊδ������2Ϊ����  
				else { CheckResult = 1;   RemoveCount++; }
				for (int z = 0; z<GrowBuffer.size(); z++)                         //����Label��¼  
				{
					CurrX = GrowBuffer.at(z).x;
					CurrY = GrowBuffer.at(z).y;
					Pointlabel.at<uchar>(CurrY, CurrX) += CheckResult;
				}
				//********�����õ㴦�ļ��**********  


			}
		}
	}

	CheckMode = 255 * (1 - CheckMode);
	//��ʼ��ת�����С������  
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
Mat histcount(Mat &img, int *ptri, uchar *ptr, int &count)   //���Ҷ�ͼ��ֱ��ͼ,ptriΪ���ͼ��Ҷȼ����������Ķ�Ӧ��ϵ����������ptrָ��ͼ���һ�����ء�
{
	int maxheight = 256;
	for (int i = 0; i < img.rows*img.cols; i++)
	{
		if (ptr[i] != 255)
		{
			++ptri[ptr[i]];
			count++;
		}
	}	                                 //ͳ��ԭͼ���Ҷȼ����ص����
	ptri[255] = 0;
	int max = maxval(ptri, 256);
	Mat hist(256, 512, CV_8UC3);                         //ֱ��ͼ��512����256
	for (int i = 0; i < 256; i++)
	{
		double height = ptri[i] * maxheight / (1.0*max);
		rectangle(hist, Point(i * 2, 255), Point((i + 1) * 2 - 1, 255 - height), Scalar(0, 0, 255));
	}
	return hist;
}
void histogramequ(Mat &img, Mat img_htg)                       //ֱ��ͼ���⺯�����������ͼ���Сһ��
{
	int count = 0;
	uchar* ptr = img.ptr<uchar>(0);
	uchar* ptr_htg = img_htg.ptr<uchar>(0);                   //���Դ��ԭͼ���Ҷȼ����ص����
	int grayori[256] = { 0 };
	double  grayp[256] = { 0 };
	int graynew[256] = { 0 };                                //����ֱ��ͼ�������Ҷȼ�ӳ���ϵ
	Mat histori/*(256, 512, CV_8UC3)*/;
	histori = histcount(img, grayori, ptr, count);
	//	imshow("hist", histori);                                //��ԭͼ��ֱ��ͼ
	for (int i = 0; i < 256; i++)
		grayp[i] = grayori[i] / (1.0*count);               //ԭͼ���Ҷȼ����ص����ռ�����ظ�������
	double sum = 0;
	int sumi;
	for (int i = 0; i < 256; i++)
	{
		sum += grayp[i];
		sumi = int(255 * sum + 0.5);

		graynew[i] = sumi;                      //����ӳ���ϵ
	}
	Mat hist_htg/*(256, 512, CV_8UC3)*/;
	for (int i = 0; i < img.rows*img.cols; i++)        //ֱ��ͼ����
	{
		if (ptr_htg[i] != 255)
		{
			ptr_htg[i] = graynew[ptr[i]];
		}
	}
	count = 0;
	memset(graynew, 0, sizeof(int)* 256);               //���洢ӳ���ϵ�������������Դ�ž����ͼ���ֱ��ͼͳ������
	hist_htg = histcount(img_htg, graynew, ptr_htg, count);
	//	imshow("hist_htg", hist_htg);                     //�����ͼ��ֱ��ͼ
}
Mat BackgroundEqu(Mat &srcImage)                                  //���������
{
	const int subWindowSize = 8;	//�Ӵ��ڴ�С
	//�Ӵ��ڻҶȾ�ֵ
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

	long long imgIntensity = 0;			//ȫͼ�ҶȾ�ֵ
	for (auto i = 0; i < srcImage.rows*srcImage.cols; ++i)
	{
		uchar *ptrSrc = srcImage.ptr<uchar>(0);
		imgIntensity += *(ptrSrc+i);
	}
	imgIntensity /= (srcImage.cols * srcImage.rows);

	for (auto i = 0; i < subMaskIntensity.rows; ++i)		//����������
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
	vector<int> statistic;			//��ֱ����ͳ��
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
//			cout << counter << ",";		//���ͳ������
			counter = 0;
		}
	}
	cout << endl;
	for (auto i = 1; i < statistic.size(); ++i)
	{
		float ForwardRatio = statistic[i] * 1.0 / (statistic[i - 1]+1);
		if (ForwardRatio > 4)		//�ж��ѷ����ֵ
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
			//	if (statistic[j] < 155) break;		//�ѷ������ֵ
			//}
			if ((j - i > 3) && (j - i < 8))          //��һ�����򴰿��ж�
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
					cout << endl << "�ѷ���" << i * 10 << "�е�" << j * 10 << "��֮��" << endl;
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
	//string imgname = "D:\\G disc\\��ҵ�Ӿ����\\���11.13\\��\\19.1.bmp";
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
