////////////////////////////////////////////////////////////////////////////////
///Copyright(c)     UESTC ROBOMASTER2018      Model Code for Mark
///ALL RIGHTS RESERVED
///@file: mark_LED.cpp
///@brief: 数码管识别源文件
///@vesion 2.9(版本号)
///@author: csm
///@email: 448554615@qq.com
///@date: 18-7-29
////////////////////////////////////////////////////////////////////////////////
#include "mark_LED.h"
//#define DEBUG
//#define OUTPUT

using namespace std;
using namespace cv;

int Mark_LED::init(string filename)
{
	//cerr << "Configure file: " << filename << endl;
	FileStorage f(filename, FileStorage::READ);

	f["FILTER_SIZE_WHOLE_LED"] >> FILTER_SIZE_WHOLE;
	f["THRESHOLD_WHOLE_LED"] >> THRESHOLD_WHOLE;
	f["ELEMENT_SIZE_WIDTH_WHOLE_LED"] >> ELEMENT_SIZE_WIDTH_WHOLE;
	f["ELEMENT_SIZE_HEIGHT_WHOLE_LED"]>>ELEMENT_SIZE_HEIGHT_WHOLE;
	f["MIN_AREA_WHOLE_LED"]>>MIN_AREA_WHOLE;
	f["MAX_AREA_WHOLE_LED"]>>MAX_AREA_WHOLE;
	f["MIN_WIDTH2HEIGHT_WHOLE_LED"]>>MIN_WIDTH2HEIGHT_WHOLE;
	f["MAX_WIDTH2HEIGHT_WHOLE_LED"]>>MAX_WIDTH2HEIGHT_WHOLE;

	f["FILTER_SIZE_SINGLE_LED"] >> FILTER_SIZE_SINGLE;
	f["THRESHOLD_SINGLE_LED"] >> THRESHOLD_SINGLE;
	f["ELEMENT_SIZE_WIDTH_SINGLE_LED"] >> ELEMENT_SIZE_WIDTH_SINGLE;
	f["ELEMENT_SIZE_HEIGHT_SINGLE_LED"]>>ELEMENT_SIZE_HEIGHT_SINGLE;
	f["MIN_AREA_SINGLE_LED"]>>MIN_AREA_SINGLE;
	f["MAX_AREA_SINGLE_LED"]>>MAX_AREA_SINGLE;

	f.release();

	for(int i=0;i<7;++i)
	{
		string name=format("..//res//LED_number_template//t%d.jpg",i);
		Mat img=imread(name,0);
		threshold(img,img,160,255,THRESH_BINARY);
		nums.emplace_back(img);
	}

	template_ahash();
	template_phash();
	template_dhash();
}

void Mark_LED::template_ahash()
{
	float average=0;
	bitset<hashLength> temp;
	Mat t;

	for(int n=0;n<7;++n)
	{
		average=0;
		resize(nums[n],t,Size(8,8));
		for(int i=0;i<8;++i)
			for(int j=0;j<8;++j)
				average+=t.at<uchar>(i,j);
		average/=64;

		for(int i=0;i<8;++i)
			for(int j=0;j<8;++j)
				temp[8*i+j]=t.at<uchar>(i,j)>=average?1:0;
		tem_ahash.emplace_back(temp);
	}
}

void Mark_LED::template_phash()
{
	float average=0;
	Mat DCT(nums[0].size(),CV_64FC1);
	bitset<hashLength> temp;	

	for(int n=0;n<7;++n)
	{	
		Mat t;
		average=0;
		resize(nums[n],t,Size(32,32));
		dct(Mat_<double>(t),DCT);
		
		for(int i=0;i<8;++i)
			for(int j=0;j<8;++j)
				average+=DCT.at<double>(i,j);
		average/=64;
		for(int i=0;i<8;++i)
			for(int j=0;j<8;++j)
				temp[8*i+j]=DCT.at<double>(i,j)>=average?1:0;
		tem_phash.emplace_back(temp);
	}
}

void Mark_LED::template_dhash()
{
	bitset<hashLength> temp;	
	Mat t;

	for(int n=0;n<7;++n)
	{	
		resize(nums[n],t,Size(8,9));
		for(int i=0;i<8;++i)
			for(int j=0;j<8;++j)
				temp[8*i+j]=t.at<uchar>(i,j)>=t.at<uchar>(i,j+1)?1:0;

		tem_dhash.emplace_back(temp);
	}
}

void Mark_LED::clear()
{
	lastPosition=Rect(0,0,0,0);
}

void Mark_LED::update()
{
	for(int i=0;i<5;++i)
	{
		number[i]=0;
		location[i]=i;
	}
}

int Mark_LED::match(Mat& LED,int method)
{
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(LED, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);
	int contour=contours.size();
	if(contour==3) 
		return 8;
	switch(method)
	{
		case 1:return match1(LED,contour);
			break;
		case 2:return match2(LED,contour);
			break;
		case 3:return match3(LED,contour);
			break;
		case 4:return match4(LED,contour);
			break;
		case 5:return match5(LED);
			break;
	}
}

int Mark_LED::match1(Mat LED,int contour)
{
	Mat diff;
	float sum=0;
	float test;

	if(contour==1)
	{
		int temp=0;
		diff=nums[0]-LED;
		for(int i=0;i<LED.rows;++i)
				for(int j=0;j<LED.cols;++j)
					test+=diff.at<uchar>(i,j)*diff.at<uchar>(i,j);
		for(int n=1;n<5;++n)
		{
			sum=0;
			diff=nums[n]-LED;
			for(int i=0;i<LED.rows;++i)
				for(int j=0;j<LED.cols;++j)
					sum+=diff.at<uchar>(i,j)*diff.at<uchar>(i,j);
			if(sum<test) {temp=n;test=sum;}
			//cout<<correspond[n]<<":"<<sum<<endl;
		}
		return correspond[temp];
	}
	else if(contour==2)
	{
		int temp=5;
		diff=nums[5]-LED;
		for(int i=0;i<LED.rows;++i)
				for(int j=0;j<LED.cols;++j)
					test+=diff.at<uchar>(i,j)*diff.at<uchar>(i,j);

		sum=0;
		diff=nums[6]-LED;
		for(int i=0;i<LED.rows;++i)
			for(int j=0;j<LED.cols;++j)
				sum+=diff.at<uchar>(i,j)*diff.at<uchar>(i,j);
		if(sum<test) {temp=6;test=sum;}
		//cout<<correspond[n]<<":"<<sum<<endl;
		return correspond[temp];
	}
	else return 0;
}

int Mark_LED::match2(Mat image,int contour)
{
	resize(image,image,Size(8,8));
	float average=0;

	for(int i=0;i<8;++i)
		for(int j=0;j<8;++j)
			average+=image.at<uchar>(i,j);
	average/=64;

	bitset<hashLength> ahash;
	for(int i=0;i<8;++i)
		for(int j=0;j<8;++j)
			ahash[8*i+j]=image.at<uchar>(i,j)>=average?1:0;

	int distance,min_distance=64;
		
	if(contour==1)
	{
		int temp=0;
		for(int n=0;n<5;++n)
		{
			distance=0;
			for(int i=0;i<hashLength;++i)
				distance+=(ahash[i]==tem_ahash[n][i]?0:1);
			//cout<<"distance:"<<distance<<endl;
			if(distance<min_distance)
			{
				min_distance=distance;
				temp=n;
			}
		}
		return correspond[temp];
	}
	else if(contour==2)
	{
		int temp=5;
		for(int n=5;n<7;++n)
		{
			distance=0;
			for(int i=0;i<hashLength;++i)
				distance+=(ahash[i]==tem_ahash[n][i]?0:1);
			//cout<<"distance:"<<distance<<endl;
			if(distance<min_distance)
			{
				min_distance=distance;
				temp=n;
			}
		}
		return correspond[temp];
	}
	else return 0;	
}

int Mark_LED::match3(Mat image,int contour)
{
	resize(image,image,Size(32,32));
	Mat DCT(image.size(),CV_64FC1);
	dct(Mat_<double>(image),DCT);
	float average=0;
	for(int i=0;i<8;++i)
		for(int j=0;j<8;++j)
			average+=DCT.at<double>(i,j);
	average/=64;

	bitset<hashLength> phash;
	for(int i=0;i<8;++i)
		for(int j=0;j<8;++j)
			phash[8*i+j]=DCT.at<double>(i,j)>=average?1:0;
	
	int distance,min_distance=64;

	if(contour==1)
	{
		int temp=0;
		for(int n=0;n<5;++n)
		{
			distance=0;
			for(int i=0;i<hashLength;++i)
				distance+=(phash[i]==tem_phash[n][i]?0:1);
			//cout<<"distance:"<<distance<<endl;
			if(distance<min_distance)
			{
				min_distance=distance;
				temp=n;
			}
		}
		return correspond[temp];
	}
	else if(contour==2)
	{	
		int temp=5;
		for(int n=5;n<7;++n)
		{
			distance=0;
			for(int i=0;i<hashLength;++i)
				distance+=(phash[i]==tem_phash[n][i]?0:1);
			//cout<<"distance:"<<distance<<endl;
			if(distance<min_distance)
			{
				min_distance=distance;
				temp=n;
			}
		}
		return correspond[temp];
	}
	else return 0;
}

int Mark_LED::match4(Mat image,int contour)
{
	resize(image,image,Size(8,9));
	bitset<hashLength> dhash;

	for(int i=0;i<8;++i)
		for(int j=0;j<8;++j)
			dhash[8*i+j]=image.at<uchar>(i,j)>=image.at<uchar>(i,j+1)?1:0;
	
	int distance,min_distance=64;

	if(contour==1)
	{
		int temp=0;
		for(int n=0;n<5;++n)
		{
			distance=0;
			for(int i=0;i<hashLength;++i)
				distance+=(dhash[i]==tem_dhash[n][i]?0:1);
			//cout<<"distance:"<<distance<<endl;
			if(distance<min_distance)
			{
				min_distance=distance;
				temp=n;
			}
		}
		return correspond[temp];	
	}
	else if(contour==2)
	{
		int temp=5;
		for(int n=5;n<7;++n)
		{
			distance=0;
			for(int i=0;i<hashLength;++i)
				distance+=(dhash[i]==tem_dhash[n][i]?0:1);
			//cout<<"distance:"<<distance<<endl;
			if(distance<min_distance)
			{
				min_distance=distance;
				temp=n;
			}
		}
		return correspond[temp];	
	}
	else return 0;
}

int Mark_LED::match5(Mat image)
{
	int tube=0;
	int tubo_roi[7][4]=
	{
		{ image.rows*0/3,image.rows*1/3,image.cols*1/2,image.cols*1/2 },
		{ image.rows*1/3,image.rows*1/3,image.cols*2/3,image.cols-1   },
		{ image.rows*2/3,image.rows*2/3,image.cols*2/3,image.cols-1   },
		{ image.rows*2/3,image.rows-1  ,image.cols*1/2,image.cols*1/2 },
		{ image.rows*2/3,image.rows*2/3,image.cols*0/3,image.cols*1/3 },
		{ image.rows*1/3,image.rows*1/3,image.cols*0/3,image.cols*1/3 },
		{ image.rows*1/3,image.rows*2/3,image.cols*1/2,image.cols*1/2 },
	};

	if(image.rows/image.cols>2.5)
		tube=6;
	else 
		for(int i=0;i<7;++i)
		{
			if(Iswhite(image,tubo_roi[i][0],tubo_roi[i][1],tubo_roi[i][2],tubo_roi[i][3]))
				tube=tube+(int)pow(2,i);
		}
	
	switch(tube)
	{
		case   6:return 1;break;
		case  91:return 2;break;
		case  79:return 3;break;
		case 102:return 4;break;
		case 109:return 5;break;
		case 125:return 6;break;
		case   7:return 7;break;
		case 127:return 8;break;
		case 111:return 9;break;
		//
		case  15:return 7;break;
		default:return -1;
	}
}

bool Mark_LED::Iswhite(Mat& image,int row0,int row1,int col0,int col1)
{
	int sum=0;
	if(row0==row1)
		for(int i=col0;i<col1;++i)
			sum+=image.at<uchar>(row0,i);
	else 
		if(col0==col1)
			for(int i=row0;i<row1;++i)
				sum+=image.at<uchar>(i,col0);
		else
			return false;

	if(sum>500)
		return true;
	else
		return false;
}
			
int Mark_LED::getLED(Mat& srcImage)
{
	vector<Mat> channels;
	Mat B, G, R, temp,ttt,GRAY;

	Rect r;
	int h=srcImage.rows;
	int w=srcImage.cols;
	if(lastPosition.area()==0)
	{
		r=Rect(0,0,w,4*h/7);
		temp=srcImage.clone();
		temp=temp(r);
	}
	else
	{
		Rect tempr;
		float a,b,c,d;
		tempr = lastPosition - Point(lastPosition.width * 0.25, lastPosition.height * 0.25)+ Size(lastPosition.width *0.5, lastPosition.height *0.5);
		a=tempr.tl().x<0?0:tempr.tl().x;
		b=tempr.tl().y<0?0:tempr.tl().y;
		c=tempr.br().x>w?w:tempr.br().x;
		d=tempr.br().y>h?h:tempr.br().y;
		r=Rect(a,b,c-a,d-b);
		temp=srcImage.clone();
		temp=temp(r);
	}

	cvtColor(temp,GRAY,COLOR_BGR2GRAY);
	split(temp, channels);
	B = channels.at(0);
	G = channels.at(1);
	R = channels.at(2);

	temp=R-G;
	//cvtColor(srcImage,temp,COLOR_BGR2GRAY);
	//imshow("R",R);
	//imshow("G",G);
	//imshow("temp",temp);
	blur(temp,temp,Size(FILTER_SIZE_WHOLE,FILTER_SIZE_WHOLE));
	//imshow("filter", temp);
	threshold(temp, temp, THRESHOLD_WHOLE, 255, THRESH_BINARY);
	//imshow("thresh", temp);

	Mat element = getStructuringElement(MORPH_RECT, Size(ELEMENT_SIZE_WIDTH_WHOLE,ELEMENT_SIZE_HEIGHT_WHOLE));
	morphologyEx(temp, temp, MORPH_CLOSE, element);
	#ifdef DEBUG
	namedWindow("LED",WINDOW_NORMAL);
   	imshow("LED",temp);
	waitKey(1);
	#endif

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	findContours(temp, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	float width, height;
	float width2height;//长宽比
	float area;//区域面积

	Rect rect;
	RotatedRect rotaRect;
	Rect LEDs;
	vector<RotatedRect> RotaLEDs;
	vector<int> sign;//记录存入RotaLEDs的轮廓序数下标

	for (int i = 0; i < contours.size(); ++i)
	{
		area = contourArea(contours[i], false);
		//cout<<"LEDarea:"<<area<<endl;
		if (area>MIN_AREA_WHOLE&&area<MAX_AREA_WHOLE)
		{
			rotaRect=minAreaRect(contours[i]);	
			float angle=rotaRect.angle;
			width=rotaRect.size.width;
			height=rotaRect.size.height;
		
			width2height = width / height;
			//cout<<"LEDangle:"<<angle<<endl;
			if((angle>=-90&&angle<=-65)||(angle>=-25&&angle<=0))
			{
				if(angle<=-65)
					width2height=height/width;
				//cout<<"LEDwidth2height:"<<width2height<<endl;
				if (width2height > MIN_WIDTH2HEIGHT_WHOLE&&width2height < MAX_WIDTH2HEIGHT_WHOLE)		
				{
					RotaLEDs.emplace_back(rotaRect);
					sign.emplace_back(i);
				}
			}
		}	
	}
	if(RotaLEDs.size()>5||RotaLEDs.size()==0) 
	{
		cout<<"LED area search error! LEDs = "<<RotaLEDs.size()<<endl;
		return 1;
	}
	for(int j=0;j<RotaLEDs.size();++j)
	{
		LEDs=boundingRect(contours[sign[j]]);
		LED_bottom_position=LEDs.br().y+r.tl().y;
		LED_left_position=LEDs.tl().x+r.tl().x;
		LED_right_position=LEDs.br().x+r.tl().x;

		lastPosition=Rect(LED_left_position,LED_bottom_position-LEDs.height,LEDs.width,LEDs.height);

		int lw,lh;
		lw=LEDs.width;
		lh=LEDs.height;
		if((LEDs.tl().x-lw*0.02>0)&&(LEDs.br().x + lw * 0.02 < GRAY.cols))
		{
			LEDs = LEDs - Point(lw * 0.02, 0);
			LEDs = LEDs + Size(lw * 0.04, 0);
		}
		if((LEDs.tl().y-lh*0.01>0)&&(LEDs.br().y + lh * 0.01 < GRAY.rows))
		{
			LEDs = LEDs - Point(0, lh * 0.01);
			LEDs = LEDs + Size(0, lh * 0.02);
		}

		temp=GRAY.clone();
		temp=temp(LEDs);
		/*if(RotaLEDs[j].angle>=-88&&RotaLEDs[j].angle<=-2)
		{
			//Point center(RotaLEDs[j].center.x-LEDs.tl().x,RotaLEDs[j].center.y-LEDs.tl().y);
			Point center(LEDs.width/2,LEDs.height/2);

			Mat map_matrix;
			if(RotaLEDs[j].angle<=-70)				
				map_matrix=getRotationMatrix2D(center,90+RotaLEDs[j].angle,1);
			else map_matrix=getRotationMatrix2D(center,RotaLEDs[j].angle,1);
			warpAffine(temp,temp,map_matrix,temp.size(),INTER_LINEAR|WARP_FILL_OUTLIERS);
		}*/
		
		//imshow("temp1",temp);
		blur(temp,temp,Size(FILTER_SIZE_SINGLE,FILTER_SIZE_SINGLE));
		//imshow("temp2",temp);
		threshold(temp,temp,THRESHOLD_SINGLE,255,THRESH_BINARY);
		Mat element2 = getStructuringElement(MORPH_RECT, Size(ELEMENT_SIZE_WIDTH_SINGLE,ELEMENT_SIZE_HEIGHT_SINGLE));
		morphologyEx(temp, temp, MORPH_CLOSE, element2);
		threshold(temp,temp,150,255,THRESH_BINARY);
		#ifdef DEBUG
		namedWindow("LED_NUM",WINDOW_NORMAL);
	   	imshow("LED_NUM",temp);
		waitKey(1);
		#endif

		vector<vector<Point>> contours2;
		findContours(temp,contours2,hierarchy,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE);
		int k=0;
		for (int i = 0; i < contours2.size(); ++i)
		{
			area = contourArea(contours2[i], false);
			//cout<<"area:"<<area<<endl;
			if (area>MIN_AREA_SINGLE&&area<MAX_AREA_SINGLE)
			{
				rect = boundingRect(contours2[i]);
				height = abs((float)rect.tl().y - (float)rect.br().y);
				width = abs((float)rect.tl().x - (float)rect.br().x);
				width2height = width / height;
				//cout<<"width2height:"<<width2height<<endl;
				if(width2height>0.15&&width2height<0.97)
				{
					if(k==5) {k=6;break;}
					
					Mat t = Mat::zeros(25,17, CV_8U);
					resize(temp(rect), t, t.size());
					LED[k] = t;
					position[k]=rect.tl().x;				
					if(width2height>0.15 && width2height< 0.4) number[k]=1;
					++k;
				}
			}
		}
		//cout<<"LED k:"<<k<<endl;
		if(k==5) 
		{
			LEDsize=contourArea(contours[sign[j]], false);
			return 0;
		}
	}
	cout<<"LED segment error!"<<endl;
	return 1;
}

void Mark_LED::LEDsort()
{
	int sign,temp;

	vector<pair<float,int>> pairs;
	for(int i=0;i<5;++i)
		pairs.emplace_back(make_pair(position[i],i));
	sort(pairs.begin(),pairs.end(),[](const pair<float,int>& p1,const pair<float,int>& p2){return p1.first<p2.first;});

	for(int i=0;i<5;++i)
		location[i]=pairs[i].second;
}


int Mark_LED::run(Mat& srcImage,int num[5])
{
	update();
	if(getLED(srcImage)) {clear();return 1;}

	LEDsort();
	int t;

	int usedNumber[9]={0,0,0,0,0,0,0,0,0};
	for(int i=0;i<5;++i)
	{
		int match_times[8]={0,0,0,0,0,0,0,0};
		t=location[i];

		//imshow("ttt", LED[t]);
		#ifdef OUTPUT
		if(number[t]==1) 
		{
			num[i]=1;
			if(usedNumber[0]==0)
				usedNumber[0]=1;
			else
			{
				cout<<"LED number recognize error!"<<endl;
				return 1;
			}
			cout<<"match:1"<<" ";
			continue;
		}
		#else
		if(number[t]==1) 
		{
			num[i]=1;
			if(usedNumber[0]==0)
				usedNumber[0]=1;
			else
				return 1;
			continue;
		}
		#endif

		//cout <<"match1:" << match(LED[t],1) << " ";
		//cout <<"match2:" << match(LED[t],2) << " ";
		//cout <<"match3:" << match(LED[t],3) << " ";
		//cout <<"match4:" << match(LED[t],4) << " ";
		//cout <<"match5:" << match(LED[t],5) << " " << endl;

		int a,b;
		a=match(LED[t],1);
		b=match(LED[t],2);
		if(a==0) {cout<<"LED number recognize error!"<<endl;return 1;}
		if(a==b) 
		{
			num[i]=a;
			if(usedNumber[a-1]==0)
				usedNumber[a-1]=1;
			else
			{
				cout<<"LED number recognize error!"<<endl;
				return 1;
			}
		}		
		else
		{
			++match_times[a-2];
			++match_times[b-2];
			++match_times[match(LED[t],3)-2];
			++match_times[match(LED[t],4)-2];
			if(match(LED[t],5)!=-1)
				++match_times[match(LED[t],5)-2];

			int max_match=0,temp;
			for(int j=0;j<8;++j)
				if(match_times[j]>max_match)
				{
					temp=j+2;
					max_match=match_times[j];
				}

			#ifdef OUTPUT
			if(max_match<3) {cout<<"LED number recognize error!"<<endl;return 1;}
			#else
			if(max_match<3) return 1;
			#endif

			num[i]=temp;
			if(usedNumber[temp-1]==0)
				usedNumber[temp-1]=1;
			else
			{
				cout<<"LED number recognize error!"<<endl;
				return 1;
			}
		}
		#ifdef OUTPUT
		cout<<"match:"<<num[i]<<" ";
		#endif

		//else 
		//return 1;
		//waitKey(0);
	}
	#ifdef OUTPUT
	cout<<endl;
	#endif
	return 0;
}

int Mark_LED::GetRegion(Mat& srcImage)
{
	vector<Mat> channels;
	Mat B, G, R, temp,ttt,GRAY;

	Rect r;
	int h=srcImage.rows;
	int w=srcImage.cols;
	if(lastPosition.area()==0)
	{
		r=Rect(0,0,w,4*h/7);
		temp=srcImage.clone();
		temp=temp(r);
	}
	else
	{
		Rect tempr;
		float a,b,c,d;
		tempr = lastPosition - Point(lastPosition.width * 0.3, lastPosition.height * 0.25)+ Size(lastPosition.width *0.6, lastPosition.height *0.5);
		a=tempr.tl().x<0?0:tempr.tl().x;
		b=tempr.tl().y<0?0:tempr.tl().y;
		c=tempr.br().x>w?w:tempr.br().x;
		d=tempr.br().y>h?h:tempr.br().y;
		r=Rect(a,b,c-a,d-b);
		temp=srcImage.clone();
		temp=temp(r);
	}

	cvtColor(temp,GRAY,COLOR_BGR2GRAY);
	split(temp, channels);
	B = channels.at(0);
	G = channels.at(1);
	R = channels.at(2);

	temp=R-G;
	//cvtColor(srcImage,temp,COLOR_BGR2GRAY);
	//imshow("R",R);
	//imshow("G",G);
	//imshow("temp",temp);
	blur(temp,temp,Size(FILTER_SIZE_WHOLE,FILTER_SIZE_WHOLE));
	//imshow("filter", temp);
	threshold(temp, temp, THRESHOLD_WHOLE, 255, THRESH_BINARY);
	//imshow("thresh", temp);

	Mat element = getStructuringElement(MORPH_RECT, Size(ELEMENT_SIZE_WIDTH_WHOLE,ELEMENT_SIZE_HEIGHT_WHOLE));
	morphologyEx(temp, temp, MORPH_CLOSE, element);
	#ifdef DEBUG
	namedWindow("LED",WINDOW_NORMAL);
   	imshow("LED",temp);
	waitKey(1);
	#endif

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	findContours(temp, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	float width, height;
	float width2height;//长宽比
	float area;//区域面积

	Rect rect;
	RotatedRect rotaRect;
	Rect LEDs;
	vector<RotatedRect> RotaLEDs;
	vector<int> sign;//记录存入RotaLEDs的轮廓序数下标

	for (int i = 0; i < contours.size(); ++i)
	{
		area = contourArea(contours[i], false);
		//cout<<"area:"<<area<<endl;
		if (area>MIN_AREA_WHOLE&&area<MAX_AREA_WHOLE)
		{
			rotaRect=minAreaRect(contours[i]);	
			float angle=rotaRect.angle;
			width=rotaRect.size.width;
			height=rotaRect.size.height;
		
			width2height = width / height;
			//cout<<"angle:"<<angle<<endl;
			if((angle>=-90&&angle<=-65)||(angle>=-25&&angle<=0))
			{
				if(angle<=-65)
					width2height=height/width;
				//cout<<"width2height:"<<width2height<<endl;
				if (width2height > MIN_WIDTH2HEIGHT_WHOLE&&width2height < MAX_WIDTH2HEIGHT_WHOLE)		
				{
					RotaLEDs.emplace_back(rotaRect);
					sign.emplace_back(i);
				}
			}
		}	
	}
	if(RotaLEDs.size()>5||RotaLEDs.size()==0) {lastPosition=Rect(0,0,0,0);cout<<"LED area search error! LEDs = "<<RotaLEDs.size()<<endl; return 1;}
	for(int j=0;j<RotaLEDs.size();++j)
	{
		LEDs=boundingRect(contours[sign[j]]);
		LED_bottom_position=LEDs.br().y+r.tl().y;
		LED_left_position=LEDs.tl().x+r.tl().x;
		LED_right_position=LEDs.br().x+r.tl().x;

		lastPosition=Rect(LED_left_position,LED_bottom_position-LEDs.height,LEDs.width,LEDs.height);

		int lw,lh;
		lw=LEDs.width;
		lh=LEDs.height;
		if((LEDs.tl().x-lw*0.02>0)&&(LEDs.br().x + lw * 0.02 < GRAY.cols))
		{
			LEDs = LEDs - Point(lw * 0.02, 0);
			LEDs = LEDs + Size(lw * 0.04, 0);
		}
		if((LEDs.tl().y-lh*0.01>0)&&(LEDs.br().y + lh * 0.01 < GRAY.rows))
		{
			LEDs = LEDs - Point(0, lh * 0.01);
			LEDs = LEDs + Size(0, lh * 0.02);
		}

		temp=GRAY.clone();
		temp=temp(LEDs);
		/*if(RotaLEDs[j].angle>=-88&&RotaLEDs[j].angle<=-2)
		{
			//Point center(RotaLEDs[j].center.x-LEDs.tl().x,RotaLEDs[j].center.y-LEDs.tl().y);
			Point center(LEDs.width/2,LEDs.height/2);

			Mat map_matrix;
			if(RotaLEDs[j].angle<=-70)				
				map_matrix=getRotationMatrix2D(center,90+RotaLEDs[j].angle,1);
			else map_matrix=getRotationMatrix2D(center,RotaLEDs[j].angle,1);
			warpAffine(temp,temp,map_matrix,temp.size(),INTER_LINEAR|WARP_FILL_OUTLIERS);
		}*/
		
		//imshow("temp1",temp);
		blur(temp,temp,Size(FILTER_SIZE_SINGLE,FILTER_SIZE_SINGLE));
		//imshow("temp2",temp);
		threshold(temp,temp,THRESHOLD_SINGLE,255,THRESH_BINARY);
		Mat element2 = getStructuringElement(MORPH_RECT, Size(ELEMENT_SIZE_WIDTH_SINGLE,ELEMENT_SIZE_HEIGHT_SINGLE));
		morphologyEx(temp, temp, MORPH_CLOSE, element2);
		threshold(temp,temp,150,255,THRESH_BINARY);
		#ifdef DEBUG
		namedWindow("LED_NUM",WINDOW_NORMAL);
	   	imshow("LED_NUM",temp);
		waitKey(1);
		#endif

		vector<vector<Point>> contours2;
		findContours(temp,contours2,hierarchy,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE);
		int k=0;
		for (int i = 0; i < contours2.size(); ++i)
		{
			area = contourArea(contours2[i], false);
			//cout<<"area:"<<area<<endl;
			if (area>MIN_AREA_SINGLE&&area<MAX_AREA_SINGLE)
			{
				rect = boundingRect(contours2[i]);
				height = abs((float)rect.tl().y - (float)rect.br().y);
				width = abs((float)rect.tl().x - (float)rect.br().x);
				width2height = width / height;
				//cout<<"width2height:"<<width2height<<endl;
				if(width2height>0.15&&width2height<0.9)
				{
					if(k==5) {k++;break;}
					++k;
				}
			}
		}
		//cout<<"LED k:"<<k<<endl;
		if(k==5) 
		{
			LEDsize=contourArea(contours[sign[j]], false);	
			return 0;
		}
	}
	cout<<"LED segment error!"<<endl;
	lastPosition=Rect(0,0,0,0);
	return 1;
}
