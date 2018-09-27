////////////////////////////////////////////////////////////////////////////////
///Copyright(c)     UESTC ROBOMASTER2018      Model Code for Mark
///ALL RIGHTS RESERVED
///@file: rect_location.cpp
///@brief: 目标定位源文件
///@vesion 2.9(版本号)
///@author: csm
///@email: 448554615@qq.com
///@date: 18-7-29
////////////////////////////////////////////////////////////////////////////////
#include "rect_location.h"
//#define OUTPUT

using namespace std;
using namespace cv;

int RectLocation::init(string filename)
{
	//cerr << "Configure file: " << filename << endl;
	FileStorage f(filename, FileStorage::READ);

	f["intrinsicMatrix720p"] >> mIntrinsicMatrix;
	f["distortionCoeffs720p"] >> mDistortionCoeffs;
	f["offsetX"] >> offset_X;
	f["offsetY"] >> offset_Y;
	f["offsetZ"] >> offset_Z;
	f["offsetPitch"] >> offsetPitch;
	f["offsetYaw"] >> offsetYaw;
	f["VELOCITY"] >> velocity;

	f.release();
}

float RectLocation::gravityKiller(float distance,float height)
{
	float v = velocity;
	float m = 2*((height*GRAVITY+v*v)-sqrt((height*GRAVITY+v*v)*(height*GRAVITY+v*v)-GRAVITY*GRAVITY*(distance*distance+height*height)))/(GRAVITY*GRAVITY);
	float time = sqrt(m);
	float beta = asin((2*height-GRAVITY*time*time)/(2*v*time))*180/PI;

	//cout<<"beta:"<<beta<<endl;
	return beta;
}

void RectLocation::GetPitchYaw(vector<Point2f>& points2d,vector<Point3f>& points3d,int box,float& pitch,float& yaw)
{	
	vector<double> object;
	solvePnP4Points(points2d,points3d,object);

	float x,y,z,x2,y2,z2;
	x=object[0];
	y=object[1];
	z=object[2];

	#ifdef OUTPUT
	cout<<"x:"<<x<<" y:"<<y<<" z:"<<z<<endl;
	#endif

	x2 = x + offset_X;
	y2 = z*sin(dipAngle) + y*cos(dipAngle) + offset_Y;
	z2 = -y*sin(dipAngle) + z*cos(dipAngle) + offset_Z;

	#ifdef OUTPUT
	cout<<"x2:"<<x2<<" y2:"<<y2<<" z2:"<<z2<<endl;
	#endif

	yaw = -(atan2(x2, z2) * 180 / PI+offsetYaw);
	pitch = gravityKiller(z2/100.0,y2/100.0)+offsetPitch;

	if(box%3==0) yaw+=-0.5;
	else if(box%3==2) yaw+=0.5;

	if(box/3==0) pitch+=0.5;
	else if(box/3==2) pitch+=-0.5;

	#ifdef OUTPUT
	cout << "pitch:" << pitch << " yaw:" << yaw << endl;
	#endif
}

void RectLocation::solvePnP4Points(vector<Point2f>& points2d,vector<Point3f>& points3d, vector<double>& object)
{	
	Mat rot = Mat::eye(3, 3, CV_64FC1);
	Mat trans = Mat::zeros(3, 1, CV_64FC1);

	//cout << "2d:" << points2d << endl;
	//cout << "3d:" << points3d << endl;

	Mat r; //旋转向量

	solvePnP(points3d,points2d,mIntrinsicMatrix,mDistortionCoeffs,r,trans);

	object = Mat_<double>(trans);
}

