////////////////////////////////////////////////////////////////////////////////
///Copyright(c)     UESTC ROBOMASTER2018      Model Code for Mark
///ALL RIGHTS RESERVED
///@file: rect_location.h
///@brief: 目标定位头文件
///@vesion 2.9(版本号)
///@author: csm
///@email: 448554615@qq.com
///@date: 18-7-29
////////////////////////////////////////////////////////////////////////////////
#ifndef SHOOT_H
#define SHOOT_H
#include <iostream>
#include "opencv2/opencv.hpp"

const double PI = 3.1415926535;
const float GRAVITY = 9.7913;

class RectLocation
{ 
private:
	float offset_X, offset_Y, offset_Z;//云台相对于相机的坐标（直接测量获得），单位为cm

	cv::Mat mIntrinsicMatrix;//相机内参
	cv::Mat mDistortionCoeffs;//相机外参

	float offsetPitch;//pitch轴偏移量
	float offsetYaw;//yaw轴偏移量

	float offset;//y轴距离偏移量

	float velocity;//子弹发射速度
	float dipAngle=-12.5*PI/180;

	/** 云台转动，瞄准
	*
	* @param: vector<Point2f>& points2d:瞄射物体在图像中的像素点位置
	* @param: Mat& rot:旋转矩阵
	* @param: Mat& trans:平移矩阵
	* @return: void
	*/
	void solvePnP4Points(std::vector<cv::Point2f>& points2d, std::vector<cv::Point3f>& points3d,std::vector<double>& object);

	float gravityKiller(float distance,float height);
public:
	int init(std::string filename);

	/** 云台转动，瞄准
	*
	* @param: vector<Point2f>& points2d:瞄射物体在图像中的像素点位置
	* @param: Mat& rot:旋转矩阵
	* @param: Mat& trans:平移矩阵
	* @param: float& pitch: pitch轴输出
	* @param: float& yaw: yaw轴输出
	* @return: void
	*/
	void GetPitchYaw(std::vector<cv::Point2f>& points2d,std::vector<cv::Point3f>& points3d,int box,float& pitch,float& yaw);
};

#endif
