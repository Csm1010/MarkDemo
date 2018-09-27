////////////////////////////////////////////////////////////////////////////////
///Copyright(c)     UESTC ROBOMASTER2018      Model Code for Mark
///ALL RIGHTS RESERVED
///@file: mark_LED.h
///@brief: 数码管识别头文件
///@vesion 2.9(版本号)
///@author: csm
///@email: 448554615@qq.com
///@date: 18-7-29
////////////////////////////////////////////////////////////////////////////////
#ifndef MARK_LED_H
#define MARK_LED_H

#include <iostream>
#include "opencv2/opencv.hpp"
#include <bitset>

#define hashLength 64

class Mark_LED
{
private:
	int FILTER_SIZE_WHOLE;
	int THRESHOLD_WHOLE;
	int ELEMENT_SIZE_WIDTH_WHOLE;
	int ELEMENT_SIZE_HEIGHT_WHOLE;
	int MIN_AREA_WHOLE;
	int MAX_AREA_WHOLE;
	float MIN_WIDTH2HEIGHT_WHOLE;
	float MAX_WIDTH2HEIGHT_WHOLE;

	int FILTER_SIZE_SINGLE;
	int THRESHOLD_SINGLE;
	int ELEMENT_SIZE_WIDTH_SINGLE;
	int ELEMENT_SIZE_HEIGHT_SINGLE;
	int MIN_AREA_SINGLE;
	int MAX_AREA_SINGLE;
//----------------------------------------------------------------------------
	cv::Rect lastPosition;//LED灯上一次的位置
	cv::Mat LED[5];//分离出的LED灯图片
	int number[5];//LED灯的预测数字(实际只标志了是否为1）
	float position[5];
	int location[5];//按位置顺序存放number中的下标
	int correspond[7]={2,3,4,5,7,6,9};

	std::vector<cv::Mat> nums;//数字模板(2,3,4,5,6,7,9)
	std::vector<std::bitset<hashLength>> tem_ahash;
	std::vector<std::bitset<hashLength>> tem_phash;
	std::vector<std::bitset<hashLength>> tem_dhash;
//----------------------------------------------------------------------------
	void template_ahash();
	void template_phash();
	void template_dhash();

	void update();

	//四种匹配方法
	int match(cv::Mat& image,int method);
	int match1(cv::Mat image,int contour);//像素差平方之和
	int match2(cv::Mat image,int contour);//平均哈希法(aHash)
	int match3(cv::Mat image,int contour);//感知哈希法(pHash)
	int match4(cv::Mat image,int contour);//差异哈希法(dHash)
	int match5(cv::Mat image);

	bool Iswhite(cv::Mat& image,int row0,int row1,int col0,int col1);

	/** 图片识别与分割
	*
	* @param: Mat& srcImage：传入的源图片，即从相机获取的未处理的图片
	* @return: int:1失败，0成功
	*/
	int getLED(cv::Mat& srcImage);

	void LEDsort();//排序
//----------------------------------------------------------------------------
public:
	int init(std::string filename);

	void clear();

	/** 获得数码管表示的数字
	*
	* @param: Mat& srcImage：传入的源图片，即从相机获取的未处理的图片
	* @param: int num[5]:数码管按顺序最后得出的数字（最终输出）
	* @return: int：1失败，0成功
	*/
	int run(cv::Mat& srcImage,int num[5]);

	int GetRegion(cv::Mat& srcImage);

	int LED_bottom_position;//LED灯底部像素
	int LED_left_position;
	int LED_right_position;

	int LEDsize;
};
#endif
