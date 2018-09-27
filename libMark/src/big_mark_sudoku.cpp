////////////////////////////////////////////////////////////////////////////////
///Copyright(c)     UESTC ROBOMASTER2018      Model Code for Mark
///ALL RIGHTS RESERVED
///@file:big_mark_sudoku.cpp
///@brief: 大神符九宫格分割识别源文件
///@vesion 2.9(版本号)
///@author: csm
///@email: 448554615@qq.com
///@date: 18-7-29
////////////////////////////////////////////////////////////////////////////////
#include "big_mark_sudoku.h"
//#define DEBUG
//#define OUTPUT

using namespace std;
using namespace cv;
using namespace tensorflow;

int BigMark_Sudoku::init(string filename,string modelPath)
{
	FileStorage f(filename, FileStorage::READ);
	f["FILTER_SIZE_BIG_SUDOKU"] >> FILTER_SIZE;
	f["THRESHOLD_BIG_SUDOKU"] >> THRESHOLD;
	f["ELEMENT_SIZE_WIDTH_BIG_SUDOKU"] >> ELEMENT_SIZE_WIDTH;
	f["ELEMENT_SIZE_HEIGHT__BIG_SUDOKU"]>>ELEMENT_SIZE_HEIGHT;
	f["MIN_AREA_BIG_SUDOKU"]>>MIN_AREA;
	f["MAX_AREA_BIG_SUDOKU"]>>MAX_AREA;
	f["MIN_WIDTH2HEIGHT_BIG_SUDOKU"]>>MIN_WIDTH2HEIGHT;
	f["MAX_WIDTH2HEIGHT_BIG_SUDOKU"]>>MAX_WIDTH2HEIGHT;

	f["WIDTH_TARGET_BIG_MARK"]>>WIDTH_TARGET;
	f["HEIGHT_TARGET_BIG_MARK"]>>HEIGHT_TARGET;

	f.release();

	if(initSession(&mSession)!=0){
		return -1;
	}
	if(loadModel(mSession,modelPath)!=0){
		return -2;
	}

	Mat black=Mat(Size(28,28),CV_8UC1,Scalar(0));
	TensorOutputInit(mSession,black);

	return 0;
}

void BigMark_Sudoku::clear()
{
	lastPosition=Rect(0,0,0,0);
}

void BigMark_Sudoku::update()
{
	for(int i=0;i<9;++i)
	{
		number[i]=0;
		location[i]=-1;
	}
}

int BigMark_Sudoku::initSession(Session **pSession)
{
	Status status = NewSession(SessionOptions(), pSession);
	if (!status.ok()) {
		cout << status.ToString() << endl;
		return 1;
	}
	return 0;
}

int BigMark_Sudoku::loadModel(Session *session, string modelPath)
{
	GraphDef graph_def;
	Status status = ReadBinaryProto(Env::Default(),modelPath, &graph_def);
	if (!status.ok()) {
		cout << status.ToString() << endl;
		return 1;
	}
	status = session->Create(graph_def);
	if (!status.ok()) {
		cout << status.ToString() << endl;
		return 1;
	}
	return 0;
}

void BigMark_Sudoku::TensorOutputInit(Session* session,Mat& img)
{
	//模型预测
	Tensor x(DT_FLOAT,TensorShape({1,784}));//定义输入张量
	vector<float>ImageData;//定义输入数据

	//获取图片像素
	for(int i=0;i<img.rows;++i)
		for(int j=0;j<img.cols;++j)
			ImageData.emplace_back((float)((img.at<uchar>(i,j))/255.0));
	auto dst=x.flat<float>().data();
	copy_n(ImageData.begin(),784,dst);//复制	

	vector<pair<string,Tensor>> inputs={{"input",x}};//定义模型输入
	vector<Tensor> outputs;//定义模型输出

	Status status=session->Run(inputs,{"output"},{},&outputs);//调用模型
}

void BigMark_Sudoku::GetTensorOutput(Session* session,Mat& img,double result[9])
{
	//模型预测
	Tensor x(DT_FLOAT,TensorShape({1,784}));//定义输入张量
	vector<float>ImageData;//定义输入数据

	//获取图片像素
	for(int i=0;i<img.rows;++i)
		for(int j=0;j<img.cols;++j)
			ImageData.emplace_back((float)((img.at<uchar>(i,j))/255.0));
	auto dst=x.flat<float>().data();
	copy_n(ImageData.begin(),784,dst);//复制	

	vector<pair<string,Tensor>> inputs={{"input",x}};//定义模型输入
	vector<Tensor> outputs;//定义模型输出

	Status status=session->Run(inputs,{"output"},{},&outputs);//调用模型
	//输出节点名为output，结果保存在outputs中
	if(!status.ok())
	{
		cout<<status.ToString()<<endl;
		return ;
	}
	//获取输出中1-9最大的可能性，从而得到输出数字
	Tensor t=outputs[0];
	int ndim=t.shape().dims();//获取张量维度
	auto tmap=t.tensor<float,2>();

	for(int i=0;i<9;++i)
		result[i]=tmap(0,i);
}

int BigMark_Sudoku::GetSudoku(Mat& srcImage)
{
	Mat B,G,R,temp,Gray;

	Rect r(sudoku_left_position,sudoku_top_position,sudoku_right_position-sudoku_left_position,sudoku_bottom_position-sudoku_top_position);
	temp=srcImage(r);

	cvtColor(temp,Gray,COLOR_BGR2GRAY);
	vector<Mat> channels;
	split(temp,channels);
	B=channels.at(0);
	G=channels.at(1);
	R=channels.at(2);
	Mat temp2=G-B;

	//imshow("B",B);
	//imshow("G",G);
	//imshow("R",R);
	//imshow("Gray",Gray);
	//imshow("temp",temp);

	//模糊和阈值
	//GaussianBlur(temp, temp, Size(FILTER_SIZE,FILTER_SIZE), 0);
	blur(Gray,temp,Size(FILTER_SIZE,FILTER_SIZE));
	//imshow("filter",temp);
	threshold(temp, temp, THRESHOLD, 255, THRESH_BINARY);
   	//imshow("threshold",temp);

	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(ELEMENT_SIZE_WIDTH,ELEMENT_SIZE_HEIGHT));
	morphologyEx(temp, temp, MORPH_CLOSE,element);
	#ifdef DEBUG
	namedWindow("sudoku1",WINDOW_NORMAL);
   	imshow("sudoku1",temp);
	waitKey(1);
	#endif

	//区域分离，找出轮廓
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(temp, contours, hierarchy, RETR_EXTERNAL,CHAIN_APPROX_SIMPLE);

	//筛选
	float width,height;
	float width2height ;//长宽比
	float area;//区域面积
	
	Rect rect;
	vector<Rect> rects;	

	float top=temp.rows;
	for (int i = 0; i < contours.size(); ++i)
	{	
		rect=boundingRect(contours[i]);
		if(rect.tl().x==0||rect.br().x==temp.cols) 
		{
			if(rect.tl().x==0&&rect.br().x==temp.cols&&rect.tl().y<top&&rect.tl().y!=0) 
			{ 
				top=rect.tl().y;
				Rect r2(0,0,temp.cols,top);
				temp=temp2(r2);
			}
			else 
				continue;
		}
		if(rect.tl().y==0||rect.br().y==temp.rows)
			continue;

		area = contourArea(contours[i], false);
		#ifdef OUTPUT
		cout<<"area:"<<area<<endl;
		#endif
		if ((area > LEDsize/11&&area < LEDsize/2||LEDsize==0)&&area>MIN_AREA&&area<MAX_AREA)
		{
			width=abs(rect.br().x-rect.tl().x);
			height=abs(rect.br().y-rect.tl().y);			

			width2height = width / height;
			#ifdef OUTPUT
			cout<<"width2height:"<<width2height<<endl;
			#endif
			if (width2height > MIN_WIDTH2HEIGHT&&width2height < MAX_WIDTH2HEIGHT)
				rects.emplace_back(rect);
		}
	}

	int k=rects.size();
	if(k<9||k>15)
	{
		if(k<9) {cout<<"again! k1="<<k<<endl;}
		else if(k>15) {cout<<"again! k1="<<k<<endl;}
		rects.clear();
		blur(temp2,temp,Size(FILTER_SIZE,FILTER_SIZE));
		//imshow("filter2",temp);
		threshold(temp, temp, 7, 255, THRESH_BINARY);
	   	//imshow("threshold2",temp);
		morphologyEx(temp, temp, MORPH_DILATE,getStructuringElement(MORPH_ELLIPSE, Size(5,5)));
		#ifdef DEBUG
		namedWindow("sudoku2",WINDOW_NORMAL);
	   	imshow("sudoku2",temp);
		waitKey(1);
		#endif

		findContours(temp, contours, hierarchy, RETR_EXTERNAL,CHAIN_APPROX_SIMPLE);
		for (int i = 0; i < contours.size(); ++i)
		{
			rect=boundingRect(contours[i]);
			if(rect.tl().x==0||rect.tl().y==0||rect.br().x==temp.cols||rect.br().y==temp.rows) continue;

			area = contourArea(contours[i], false);
			//cout<<"area:"<<area<<endl;
			if (area > MIN_AREA &&area < MAX_AREA)
			{
				width=abs(rect.br().x-rect.tl().x);
				height=abs(rect.br().y-rect.tl().y);			

				width2height = width / height;
				//cout<<"width2height:"<<width2height<<endl;
				if (width2height > MIN_WIDTH2HEIGHT&&width2height < MAX_WIDTH2HEIGHT)
					rects.emplace_back(rect);
			}
		}
		k=rects.size();
		if(k<9) {cout<<"again! k2="<<k<<endl; return 1;}
		else if(k>15) {cout<<"again! k2="<<k<<endl; return 1;}
	}
	if(k==9)
	{
		for(int i=0;i<9;++i)
		{
			int w=rects[i].width;
			int h=rects[i].height;

			Mat t=Gray(rects[i]);

			int top,bottom,left,right;
			if(h>w)
			{
				top=bottom=h/4;
				left=right=(h-w)/2+top;
			}
			else
			{
				left=right=w/4;
				top=bottom=(w-h)/2+left;
			}
			copyMakeBorder(t,t,top,bottom,left,right,BORDER_CONSTANT,Scalar(0));
			resize(t,t,Size(28,28));

			/*Mat t = Mat::zeros(20,20, CV_8U);
			resize(Gray(rects[i]), t, t.size());
			
			copyMakeBorder(t,t,4,4,4,4,BORDER_CONSTANT,Scalar(0));*/
			Sudoku[i]=t.clone();
			temp_rect[i]=rects[i];

			Position[i].x=(rects[i].br().x+rects[i].tl().x)/2;
			Position[i].y=(rects[i].br().y+rects[i].tl().y)/2;

			//imshow("sudoku",Sudoku[i]);
			//waitKey(0);
		}
	}
	else screen(Gray,rects);

	return 0;
}

void BigMark_Sudoku::screen(Mat& srcImage,vector<Rect>& rects)
{
	int n=rects.size();
	float distance[15][15]={0};
	for(int i=0;i<n;++i)
		for(int j=0;j<n;++j)
		{
			distance[i][j]=pow((rects[i].tl().x-rects[j].tl().x),2)+pow((rects[i].tl().y-rects[j].tl().y),2);
			distance[j][i]=distance[i][j];
		}
	
	float min_distance=100000000;
	float dist;
	int sign;
	for(int i=0;i<n;++i)
	{
		dist=0;
		for(int j=0;j<n;++j)
			dist+=distance[i][j];
		if(dist<min_distance)
		{
			min_distance=dist;
			sign=i;
		}
	}	
	
	vector<pair<float,int>> pairs;
	for(int i=0;i<n;++i)
		pairs.emplace_back(make_pair(distance[sign][i],i));
	sort(pairs.begin(),pairs.end(),[](const pair<float,int>& p1,const pair<float,int>& p2){return p1.first<p2.first;});

	int k;
	for(int i=0;i<9;++i)
	{
		k=pairs[i].second;
		Mat t=srcImage(rects[k]);
		int w=rects[k].width;
		int h=rects[k].height;
		int top,bottom,left,right;
		if(h>w)
		{
			top=bottom=h/4;
			left=right=(h-w)/2+top;
		}
		else
		{
			left=right=w/4;
			top=bottom=(w-h)/2+left;
		}
		copyMakeBorder(t,t,top,bottom,left,right,BORDER_CONSTANT,Scalar(0));
		resize(t,t,Size(28,28));

		/*Mat t = Mat::zeros(20,20, CV_8U);
		resize(srcImage(rects[k]), t, t.size());
		copyMakeBorder(t,t,4,4,4,4,BORDER_CONSTANT,Scalar(0));	*/	

		Sudoku[i]=t.clone();
		temp_rect[i]=rects[k];
	
		Position[i].x=(rects[k].br().x+rects[k].tl().x)/2;
		Position[i].y=(rects[k].br().y+rects[k].tl().y)/2;
		//imshow("Sudoku",Sudoku[k]);
		//waitKey(0);
	}
}

int BigMark_Sudoku::run(Mat& srcImage,int top_position,int left_position,int right_position,int size)
{
	LEDsize=size;
	#ifdef OUTPUT
	cout<<"LEDsize:"<<LEDsize<<endl;
	#endif
	int h=srcImage.rows;
	int w=srcImage.cols;
	if(lastPosition.area()==0)
	{
		sudoku_top_position=top_position;
		sudoku_bottom_position=h;
		sudoku_left_position=left_position;
		sudoku_right_position=right_position;
	}
	else
	{
		Rect tempr;
		float a,b,c,d;
		tempr = lastPosition - Point(lastPosition.width / 4, lastPosition.height / 3)+ Size(lastPosition.width / 2 , lastPosition.height / 3 * 2);

		sudoku_top_position=tempr.tl().y<top_position?top_position:tempr.tl().y;
		sudoku_bottom_position=tempr.br().y>h?h:tempr.br().y;
		sudoku_left_position=tempr.tl().x<0?0:tempr.tl().x;
		sudoku_right_position=tempr.br().x>w?w:tempr.br().x;
	}

	update();

	if(GetSudoku(srcImage)) {clear();return 1;}

	int temp=0;
	for(int i=0;i<9;++i)
	{
		//bitwise_not(Sudoku[i],Sudoku[i]);
		//imshow("Sudoku",Sudoku[i]);
		//waitKey(0);
		//Mat element = getStructuringElement(MORPH_ELLIPSE, Size(2,2));
		//morphologyEx(Sudoku[i], Sudoku[i], MORPH_CLOSE,element);
				
		GetTensorOutput(mSession,Sudoku[i],prob[i]);

		//cout<<"prob_"<<i<<":";
		/*for(int k=0;k<9;++k)
			cout<<prob[i][k]<<" ";
		cout<<endl;*/

		for(int j=0;j<9;++j)
			if(prob[i][j]>prob[i][temp])
				temp=j;
		number[i]=temp;
	
		//修正
		if(location[temp]!=-1)
		{
			int k=location[temp];
			if(prob[k][temp]>prob[i][temp])
				revision(k,i,temp);
			else
				revision(i,k,temp);
		}
		else 
			location[temp]=i;
	}
	return 0;
}

void BigMark_Sudoku::revision(int a,int b,int num)
{
	number[a]=num;
	location[num]=a;

	prob[b][num]=0;
	int temp=0;
	for(int i=0;i<9;++i)
		if(prob[b][i]>prob[b][temp])
			temp=i;
	number[b]=temp;
	
	if(location[temp]!=-1)
	{
		int k=location[temp];
		if(prob[k][temp]>prob[b][temp])
			revision(k,b,temp);
		else
			revision(b,k,temp);
	}
	else 
		location[temp]=b;	
}

void BigMark_Sudoku::SudokuSort(int num_Sudoku[9])
{
	int index[9];
	int temp;
	float xl,xr,yt,yb;

	vector<pair<float,int>> pairs;
	for(int i=0;i<9;++i)
		pairs.emplace_back(make_pair(Position[i].y,i));
	sort(pairs.begin(),pairs.end(),[](const pair<float,int>& p1,const pair<float,int>& p2){return p1.first<p2.first;});

	for(int i=0;i<9;++i)
		index[i]=pairs[i].second;

	yt=Position[index[0]].y+sudoku_top_position;
	yb=Position[index[8]].y+sudoku_top_position;

	int temp2;
	for(int k=0;k<3;++k)
	{
		temp2=3*k;
		for(int i=0;i<2;++i)
			for(int j=i+1;j<3;++j)
				if(Position[index[temp2+i]].x>Position[index[temp2+j]].x)
				{
					temp=index[temp2+i];
					index[temp2+i]=index[temp2+j];
					index[temp2+j]=temp;
				}	
	}

	for(int i=0;i<9;++i)
	{
		sortIndex[i]=index[i];
		revSort[index[i]]=i;
		num_Sudoku[i]=number[index[i]]+1;
	}

	int min=Position[sortIndex[0]].x;
	if(Position[sortIndex[3]].x<min)
		min=Position[sortIndex[3]].x;
	if(Position[sortIndex[6]].x<min)
		min=Position[sortIndex[6]].x;

	int max=Position[sortIndex[2]].x;	
	if(Position[sortIndex[5]].x>max)
		max=Position[sortIndex[5]].x;
	if(Position[sortIndex[8]].x>max)
		max=Position[sortIndex[8]].x;

	xl=min+sudoku_left_position;
	xr=max+sudoku_left_position;

	lastPosition=Rect(xl,yt,xr-xl,yb-yt);
}

void BigMark_Sudoku::GetPosition(int n,vector<Point2f>& points2d,vector<Point3f>& points3d,int model)
{
	float xl,xr,yt,yb;
	/*for(int i=0;i<9;++i)
		cout<<Position[i]<<" ";
	cout<<endl;
	for(int i=0;i<9;++i)
		cout<<Position[sortIndex[i]]<<" ";
	cout<<endl;*/
	yt=(Position[sortIndex[0]].y+Position[sortIndex[1]].y+Position[sortIndex[2]].y)/3+sudoku_top_position;
	yb=(Position[sortIndex[6]].y+Position[sortIndex[7]].y+Position[sortIndex[8]].y)/3+sudoku_top_position;
	xl=(Position[sortIndex[0]].x+Position[sortIndex[3]].x+Position[sortIndex[6]].x)/3+sudoku_left_position;
	xr=(Position[sortIndex[2]].x+Position[sortIndex[5]].x+Position[sortIndex[8]].x)/3+sudoku_left_position;
	points2d.emplace_back(Point2f(xl,yt));
	points2d.emplace_back(Point2f(xr,yt));
	points2d.emplace_back(Point2f(xr,yb));
	points2d.emplace_back(Point2f(xl,yb));
	
	//cout<<points2d<<endl;

	float half_x=WIDTH_TARGET/2.0;
	float half_y=HEIGHT_TARGET/2.0;

	if(model==0)
	{
		n=location[n-1];
		n=revSort[n];
	}

	int offx=n%3;
	int offy=n/3;
	points3d.emplace_back(Point3f(0-offx*half_x,0-offy*half_y,0));
	points3d.emplace_back(Point3f(WIDTH_TARGET-offx*half_x,0-offy*half_y,0));
	points3d.emplace_back(Point3f(WIDTH_TARGET-offx*half_x,HEIGHT_TARGET-offy*half_y,0));
	points3d.emplace_back(Point3f(0-offx*half_x,HEIGHT_TARGET-offy*half_y,0));
	//cout<<points3d<<endl;
}

void BigMark_Sudoku::DrawBox(int n,Mat& srcImage)
{
	for(int i=0;i<9;++i) 
	{
		temp_rect[i] = temp_rect[i] + Point(sudoku_left_position,sudoku_top_position);
		rectangle(srcImage, temp_rect[i], Scalar(255, 0, 255), 3);
	}

	int t=location[n-1];
	rectangle(srcImage,temp_rect[t],Scalar(255,255,0),3);

	//imshow("1",srcImage);
	//waitKey(1);
}

void BigMark_Sudoku::SavePicture(int count)
{
	for(int i=0;i<9;++i)
	{
		string name = format("..//res//picture//%d//%d.jpg",number[i]+1,count);
		imwrite(name, Sudoku[i]);
	}
}


