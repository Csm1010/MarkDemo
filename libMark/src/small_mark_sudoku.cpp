////////////////////////////////////////////////////////////////////////////////
///Copyright(c)     UESTC ROBOMASTER2018      Model Code for Mark
///ALL RIGHTS RESERVED
///@file: small_mark_sudoku.cpp
///@brief: 小神符九宫格分割识别源文件
///@vesion 2.9(版本号)
///@author: csm
///@email: 448554615@qq.com
///@date: 18-7-29
////////////////////////////////////////////////////////////////////////////////
#include "small_mark_sudoku.h"
//#define DEBUG
//#define OUTPUT

using namespace std;
using namespace cv;
using namespace tensorflow;

int SmallMark_Sudoku::init(string filename,string modelPath)
{
	FileStorage f(filename, FileStorage::READ);
	f["FILTER_SIZE_SMALL_SUDOKU"] >> FILTER_SIZE;
	f["THRESHOLD_SMALL_SUDOKU"] >> THRESHOLD;
	f["ELEMENT_SIZE_WIDTH_SMALL_SUDOKU"] >> ELEMENT_SIZE_WIDTH;
	f["ELEMENT_SIZE_HEIGHT__SMALL_SUDOKU"]>>ELEMENT_SIZE_HEIGHT;
	f["MIN_AREA_SMALL_SUDOKU"]>>MIN_AREA;
	f["MAX_AREA_SMALL_SUDOKU"]>>MAX_AREA;
	f["MIN_WIDTH2HEIGHT_SMALL_SUDOKU"]>>MIN_WIDTH2HEIGHT;
	f["MAX_WIDTH2HEIGHT_SMALL_SUDOKU"]>>MAX_WIDTH2HEIGHT;

	f["THRESH"] >>  THRESH2;

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

void SmallMark_Sudoku::clear()
{
	lastPosition=Rect(0,0,0,0);
}

void SmallMark_Sudoku::update()
{
	for(int i=0;i<9;++i)
	{
		number[i]=0;
		location[i]=-1;
	}
}

int SmallMark_Sudoku::initSession(Session **pSession)
{
	Status status = NewSession(SessionOptions(), pSession);
	if (!status.ok()) {
		cout << status.ToString() << endl;
		return 1;
	}
	return 0;
}

int SmallMark_Sudoku::loadModel(Session *session, string modelPath)
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

void SmallMark_Sudoku::TensorOutputInit(Session* session,Mat& img)
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

void SmallMark_Sudoku::GetTensorOutput(Mat& img,double result[9])
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

	Status status=mSession->Run(inputs,{"output"},{},&outputs);//调用模型
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

	for(int i=1;i<10;++i)
		result[i-1]=tmap(0,i);
}

int SmallMark_Sudoku::GetSudoku(Mat& srcImage0)
{
	Mat srcImage,Image;

	Rect r(sudoku_left_position,sudoku_top_position,sudoku_right_position-sudoku_left_position,sudoku_bottom_position-sudoku_top_position);
	srcImage=srcImage0(r);

	//imshow("srcI",srcImage);
	//waitKey(1);

	//模糊和阈值
	//GaussianBlur(srcImage, Image, Size(FILTER_SIZE,FILTER_SIZE), 0);
	blur(srcImage,Image,Size(FILTER_SIZE,FILTER_SIZE));
	//imshow("filter",Image);
	threshold(Image, Image, THRESHOLD, 255, THRESH_BINARY);
	//imshow("thresh",Image);

	//形态学处理
	Mat element = getStructuringElement(MORPH_RECT, Size(ELEMENT_SIZE_WIDTH,ELEMENT_SIZE_HEIGHT));
	morphologyEx(Image, Image, MORPH_CLOSE,element);
	#ifdef DEBUG
	namedWindow("Sudoku",WINDOW_NORMAL);
	imshow("Sudoku",Image);
	waitKey(1);
	#endif

	//区域分离，找出轮廓
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(Image, contours, hierarchy, RETR_LIST,CHAIN_APPROX_SIMPLE);

	//筛选
	float width,height;
	float width2height ;//长宽比
	float area;//区域面积
	float angle;//偏转角

	RotatedRect rotaRect;
	Rect rect;
	vector<RotatedRect> rotaRects;

	vector<int> sign;//记录存入rotaRects的轮廓序数下标
	for (int i = 0; i < contours.size(); ++i)
	{	
		rect=boundingRect(contours[i]);
		if(rect.tl().x==0||rect.tl().y==0||rect.br().x==srcImage.cols||rect.br().y==srcImage.rows) continue;

		area = contourArea(contours[i], false);
		#ifdef OUTPUT
		cout<<"area:"<<area<<endl;
		#endif
		if ((area > LEDsize/5&&area < LEDsize*6/5||LEDsize==0)&&area>MIN_AREA&&area<MAX_AREA)
		{
			rotaRect=minAreaRect(contours[i]);
			angle=rotaRect.angle;
			width=rotaRect.size.width;
			height=rotaRect.size.height;
		
			width2height = width / height;
			//cout<<angle<<endl;
			if((angle>=-90&&angle<=-75)||(angle>=-15&&angle<=0))
			{
				if(angle<=-75)
					width2height=height/width;
				#ifdef OUTPUT
				cout<<"width2height:"<<width2height<<endl;
				#endif
				if (width2height > MIN_WIDTH2HEIGHT&&width2height < MAX_WIDTH2HEIGHT)		
				{
					rotaRects.emplace_back(rotaRect);
					sign.emplace_back(i);
				}
			}
		}
	}

	int k=rotaRects.size();
	//cout<<"sudoku k:"<<k<<endl;
	if(k<9) {cout<<"again! sudoku k="<<k<<endl; return 1;}
	else if(k>15) {cout<<"again! sudoku k="<<k<<endl; return 1;}
	else if(k==9)
	{
		for(int i=0;i<9;++i)
		{
			Mat t = Mat::zeros(28, 28, CV_8U);
			if(rotaRects[i].angle<-85||rotaRects[i].angle>-5)
			{
				rect=boundingRect(contours[sign[i]]);
				
				/*Mat passer;

				blur(srcImage,passer,Size(5,5));

				threshold(passer,passer,THRESH2,255,THRESH_BINARY);
				//threshold(passer,passer,0,255,THRESH_OTSU);

				Mat black=Mat(srcImage.size(),CV_8UC1,Scalar(0));
				Mat white=Mat(srcImage.size(),CV_8UC1,Scalar(255));

				rectangle(black,rect,Scalar(255),-1,8);

				Mat element = getStructuringElement(MORPH_RECT, Size(11,11));
				morphologyEx(black, black, MORPH_ERODE,element);

				white=white-black;
				passer=passer+white;*/

				Mat passer=srcImage(rect);
				threshold(passer,passer,THRESH2,255,THRESH_BINARY);
				//threshold(passer,passer,0,255,THRESH_OTSU);

				int rw=rect.width;
				int rh=rect.height;
				rect = rect + Point(-rect.tl().x,-rect.tl().y);
				rect = rect + Size(-rw * 0.2, -rh * 0.2);
				rect = rect + Point(rw*0.1,rh*0.1);
				Mat box=passer(rect);
				box=box.clone();
				copyMakeBorder(box,box,rh*0.1,rh*0.1,rw*0.1,rw*0.1,BORDER_CONSTANT,Scalar(255));

				resize(box, t, t.size());
				#ifdef DEBUG
				string name=format("1-%d",i);
				imshow(name,box);
				waitKey(1);
				#endif
			}
			else
			{
				Mat mask=Mat(srcImage.size(),Image.type(),Scalar(0));
				Mat temp=Mat(srcImage.size(),Image.type(),Scalar(0));
				
				Point2f P[4];
				rotaRects[i].points(P);

				Point root_points[1][4];
				root_points[0][0]=P[0];
				root_points[0][1]=P[1];
				root_points[0][2]=P[2];
				root_points[0][3]=P[3];
				const Point* ppt[1]={root_points[0]};
				int npt[] = {4}; 
				polylines(mask,ppt,npt,1,1,Scalar(255),1,8,0);
				fillPoly(mask,ppt,npt,1,Scalar(255));

				Mat passer;
				blur(srcImage,passer,Size(5,5));
				threshold(passer,passer,THRESH2,255,THRESH_BINARY);

				#ifdef DEBUG
				imshow("passer",passer);
				waitKey(1);
				#endif

				passer.copyTo(temp,mask);

				Mat RotationedImg;
				Mat map_matrix;
				if(rotaRects[i].angle<=-75)				
					map_matrix=getRotationMatrix2D(rotaRects[i].center,90+rotaRects[i].angle,1);
				else map_matrix=getRotationMatrix2D(rotaRects[i].center,rotaRects[i].angle,1);
				warpAffine(temp,RotationedImg,map_matrix,temp.size(),INTER_LINEAR|WARP_FILL_OUTLIERS);
				
				vector<vector<Point>> cont2;
				findContours(RotationedImg,cont2,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE);
				if(cont2.size()==0) return 1;
				rect=boundingRect(cont2[0]);

				Mat black=Mat(srcImage.size(),CV_8UC1,Scalar(0));
				Mat white=Mat(srcImage.size(),CV_8UC1,Scalar(255));

				//drawContours(black,cont2,-1,Scalar(255),FILLED,8);
				rectangle(black,rect,Scalar(255),-1,8);

				Mat element = getStructuringElement(MORPH_RECT, Size(9,9));
				morphologyEx(black, black, MORPH_ERODE,element);

				white=white-black;

				RotationedImg=RotationedImg+white;
				resize(RotationedImg(rect), t, t.size());
				//imshow("R",t);
				//waitKey(1);
			}	
			
			bitwise_not(t,t);
			Sudoku[i]=t.clone();
			Position[i]=rotaRects[i];

			//imshow("t",t);
			//waitKey(0);
		}
	}
	else return screen(srcImage,rotaRects,contours,sign);
	return 0;
}

int SmallMark_Sudoku::screen(Mat& srcImage,vector<RotatedRect>& rotaRects,vector<vector<Point>>& contours,vector<int>& sign)
{
	int n=rotaRects.size();
	float distance[15][15]={0};
	for(int i=0;i<n;++i)
		for(int j=0;j<n;++j)
		{
			distance[i][j]=pow((rotaRects[i].center.x-rotaRects[j].center.x),2)+pow((rotaRects[i].center.y-rotaRects[j].center.y),2);
			distance[j][i]=distance[i][j];
		}
	
	float min_distance=100000000;
	float dist;
	int t;
	for(int i=0;i<n;++i)
	{
		dist=0;
		for(int j=0;j<n;++j)
			dist+=distance[i][j];
		if(dist<min_distance)
		{
			min_distance=dist;
			t=i;
		}
	}	
	
	vector<pair<float,int>> pairs;
	for(int i=0;i<n;++i)
		pairs.emplace_back(make_pair(distance[t][i],i));
	sort(pairs.begin(),pairs.end(),[](const pair<float,int>& p1,const pair<float,int>& p2){return p1.first<p2.first;});

	int k;
	Rect rect;
	for(int i=0;i<9;++i)
	{
		k=pairs[i].second;
		Mat t = Mat::zeros(28, 28, CV_8U);
		if(rotaRects[k].angle<-85||rotaRects[k].angle>-5)
		{
			rect=boundingRect(contours[sign[k]]);

			/*Mat passer;
			blur(srcImage,passer,Size(5,5));
			threshold(passer,passer,THRESH2,255,THRESH_BINARY);
            		//imshow("passer",passer);
            		//waitKey(1);

			Mat black=Mat(srcImage.size(),CV_8UC1,Scalar(0));
			Mat white=Mat(srcImage.size(),CV_8UC1,Scalar(255));

			//drawContours(black,contours,sign[i],Scalar(255),FILLED,8);
			rectangle(black,rect,Scalar(255),-1,8);

			Mat element = getStructuringElement(MORPH_RECT, Size(9,9));
			morphologyEx(black, black, MORPH_ERODE,element);

			white=white-black;
			passer=passer+white;*/

			Mat passer=srcImage(rect);
			threshold(passer,passer,THRESH2,255,THRESH_BINARY);

			int rw=rect.width;
			int rh=rect.height;
			rect = rect + Point(-rect.tl().x,-rect.tl().y);
			rect = rect + Size(-rw * 0.2, -rh * 0.2);
			rect = rect + Point(rw*0.1,rh*0.1);
			Mat box=passer(rect);
			box=box.clone();
			copyMakeBorder(box,box,rh*0.1,rh*0.1,rw*0.1,rw*0.1,BORDER_CONSTANT,Scalar(255));

			resize(box, t, t.size());
			#ifdef DEBUG
			string name=format("2-%d",i);
			imshow(name,box);
			waitKey(1);
			#endif
		}
		else
		{
			Mat mask=Mat(srcImage.size(),CV_8UC1,Scalar(0));
			Mat temp=Mat(srcImage.size(),CV_8UC1,Scalar(0));
			
			Point2f P[4];
			rotaRects[i].points(P);

			Point root_points[1][4];
			root_points[0][0]=P[0];
			root_points[0][1]=P[1];
			root_points[0][2]=P[2];
			root_points[0][3]=P[3];
			const Point* ppt[1]={root_points[0]};
			int npt[] = {4}; 
			polylines(mask,ppt,npt,1,1,Scalar(255),1,8,0);
			fillPoly(mask,ppt,npt,1,Scalar(255));

			Mat passer;
			blur(srcImage,passer,Size(5,5));
			threshold(passer,passer,THRESH2,255,THRESH_BINARY);

			#ifdef DEBUG
			imshow("passer",passer);
			waitKey(1);
			#endif

			passer.copyTo(temp,mask);

			Mat RotationedImg;				
			Mat map_matrix;
			if(rotaRects[i].angle<=-75)				
					map_matrix=getRotationMatrix2D(rotaRects[i].center,90+rotaRects[i].angle,1);
			map_matrix=getRotationMatrix2D(rotaRects[k].center,rotaRects[k].angle,1);
			warpAffine(temp,RotationedImg,map_matrix,temp.size(),INTER_LINEAR|WARP_FILL_OUTLIERS);
				
			vector<vector<Point>> cont2;
			findContours(RotationedImg,cont2,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE);
			if(cont2.size()==0) return 1;
			rect=boundingRect(cont2[0]);

			Mat black=Mat(srcImage.size(),CV_8UC1,Scalar(0));
			Mat white=Mat(srcImage.size(),CV_8UC1,Scalar(255));

			//drawContours(black,cont2,-1,Scalar(255),FILLED,8);
			rectangle(black,rect,Scalar(255),-1,8);

			Mat element = getStructuringElement(MORPH_RECT, Size(9,9));
			morphologyEx(black, black, MORPH_ERODE,element);

			white=white-black;

			RotationedImg=RotationedImg+white;
			resize(RotationedImg(rect), t, t.size());
		}
		
		bitwise_not(t,t);		
		Sudoku[i]=t.clone();
		Position[i]=rotaRects[k];
	}
	
}

int SmallMark_Sudoku::run(Mat& srcImage,int top_position,int left_position,int right_position,int size)
{	
	Mat Image;
	LEDsize=size;
	#ifdef OUTPUT
	cout<<"LEDsize:"<<LEDsize<<endl;
	#endif
	if(srcImage.channels()==3)
		cvtColor(srcImage,Image,COLOR_BGR2GRAY);

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
		tempr = lastPosition - Point(lastPosition.width / 3, lastPosition.height / 3)+ Size(lastPosition.width / 3 * 2 , lastPosition.height / 3 *2);

		sudoku_top_position=tempr.tl().y<top_position?top_position:tempr.tl().y;
		sudoku_bottom_position=tempr.br().y>h?h:tempr.br().y;
		sudoku_left_position=tempr.tl().x<0?0:tempr.tl().x;
		sudoku_right_position=tempr.br().x>w?w:tempr.br().x;
	}

	update();

	if(GetSudoku(Image)) {clear();return 1;}

	int temp=0;
	for(int i=0;i<9;++i)
	{
		GetTensorOutput(Sudoku[i],prob[i]);

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

void SmallMark_Sudoku::revision(int a,int b,int num)
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

void SmallMark_Sudoku::SudokuSort(int num_sudoku[9])
{
	int index[9];
	int temp;
	float xl,xr,yt,yb;

	vector<pair<float,int>> pairs;
	for(int i=0;i<9;++i)
		pairs.emplace_back(make_pair(Position[i].center.y,i));
	sort(pairs.begin(),pairs.end(),[](const pair<float,int>& p1,const pair<float,int>& p2){return p1.first<p2.first;});

	for(int i=0;i<9;++i)
		index[i]=pairs[i].second;

	yt=Position[index[0]].center.y+sudoku_top_position;
	yb=Position[index[8]].center.y+sudoku_top_position;


	int temp2;
	for(int k=0;k<3;++k)
	{
		temp2=3*k;
		for(int i=0;i<2;++i)
			for(int j=i+1;j<3;++j)
				if(Position[index[temp2+i]].center.x>Position[index[temp2+j]].center.x)
				{
					temp=index[temp2+i];
					index[temp2+i]=index[temp2+j];
					index[temp2+j]=temp;
				}	
	}

	for(int i=0;i<9;++i)
	{
		sortIndex[i] = index[i];
		num_sudoku[i] = number[index[i]] + 1;
	}

	int min=Position[sortIndex[0]].center.x;
	if(Position[sortIndex[3]].center.x<min)
		min=Position[sortIndex[3]].center.x;
	if(Position[sortIndex[6]].center.x<min)
		min=Position[sortIndex[6]].center.x;

	int max=Position[sortIndex[2]].center.x;	
	if(Position[sortIndex[5]].center.x>max)
		max=Position[sortIndex[5]].center.x;
	if(Position[sortIndex[8]].center.x>max)
		max=Position[sortIndex[8]].center.x;

	xl=min+sudoku_left_position;
	xr=max+sudoku_left_position;

	lastPosition=Rect(xl,yt,xr-xl,yb-yt);
}

void SmallMark_Sudoku::GetPosition(int n,vector<Point2f>& points2d,int model)
{
	if(model==0) 
	{
		n = location[n - 1];
	}
	else n=sortIndex[n];

	RotatedRect rotaRect = Position[n];

	Point2f P[4];
	rotaRect.points(P);
	
	int sign[4]={0,1,2,3};
	int min=0,sec_min=1,a=2,b=3;
	int temp1,temp2;

	for(int i=0;i<4;++i)
	{
		P[i].y+=sudoku_top_position;
		P[i].x+=sudoku_left_position;
	}

	if(P[0].x>P[1].x) {min=1;sec_min=0;}

	if(P[a].x<P[sec_min].x)
	{
		if(P[a].x<P[min].x)
		{
			temp1=min;
			min=a;
			temp2=sec_min;
			sec_min=temp1;
			a=temp2;
		}
		else
		{
			temp1=sec_min;
			sec_min=a;
			a=temp1;
		}
	}

	if(P[b].x<P[sec_min].x)
	{
		if(P[b].x<P[min].x)
		{
			temp1=min;
			min=b;
			temp2=sec_min;
			sec_min=temp1;
			b=temp2;
		}
		else
		{
			temp1=sec_min;
			sec_min=b;
			b=temp1;
		}
	}

	if(P[min].y<P[sec_min].y)
	{
		sign[0]=min;
		sign[3]=sec_min;
	}
	else
	{
		sign[0]=sec_min;
		sign[3]=min;
	}

		
	if(P[a].y<P[b].y)
	{
		sign[1]=a;
		sign[2]=b;
	}
	else
	{
		sign[1]=b;
		sign[2]=a;
	}	

	//cout<<min<<" "<<sec_min<<" "<<a<<" "<<b<<endl;

	for(int i=0;i<4;++i)
		points2d.emplace_back(P[sign[i]]);
}

void SmallMark_Sudoku::DrawBox(int n,Mat& srcImage)
{
	RotatedRect rotaRect;
	Point2f P[4];
	for(int i=0;i<9;++i)
	{
		rotaRect=Position[i];
		rotaRect.points(P);
		for(int k=0;k<4;++k)
		{
			P[k].y+=sudoku_top_position;
			P[k].x+=sudoku_left_position;
		}
		for(int j=0;j<4;++j)
			line(srcImage,P[j],P[(j+1)%4],Scalar(255,0,255),3);
	}

	int t=location[n-1];
	rotaRect=Position[t];
	rotaRect.points(P);

	for(int i=0;i<4;++i)
	{
		P[i].y+=sudoku_top_position;
		P[i].x+=sudoku_left_position;
	}
	for(int i=0;i<4;++i)
		line(srcImage,P[i],P[(i+1)%4],Scalar(255,255,0),3);

	//imshow("1",srcImage);
	//waitKey(1);
}

void SmallMark_Sudoku::SavePicture(int count)
{
	for(int i=0;i<9;++i)
	{
		string name = format("..//res//picture//%d//%d.jpg",number[i]+1,count);
		imwrite(name, Sudoku[i]);
	}
}

