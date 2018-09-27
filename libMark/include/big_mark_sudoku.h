////////////////////////////////////////////////////////////////////////////////
///Copyright(c)     UESTC ROBOMASTER2018      Model Code for Mark
///ALL RIGHTS RESERVED
///@file:big_mark_sudoku.h
///@brief: 大神符九宫格分割识别头文件
///@vesion 2.9(版本号)
///@author: csm
///@email: 448554615@qq.com
///@date: 18-7-29
////////////////////////////////////////////////////////////////////////////////
#ifndef BIGMARK_SUDOKU_H
#define BIGMARK_SUDOKU_H

#include <iostream>
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "opencv2/opencv.hpp"

class BigMark_Sudoku
{
private:
	int FILTER_SIZE;
	int THRESHOLD;
	int ELEMENT_SIZE_WIDTH;
	int ELEMENT_SIZE_HEIGHT;
	int MIN_AREA;
	int MAX_AREA;
	float MIN_WIDTH2HEIGHT;
	float MAX_WIDTH2HEIGHT;

	float WIDTH_TARGET;
	float HEIGHT_TARGET;

	int LEDsize;
//---------------------------------------------------------------------------
	cv::Mat Sudoku[9];//分离出的九宫格图片
	double prob[9][9];//九宫格通过模型预测出的每个格子为1-9的概率
	int number[9]={0};//九宫格的预测数字-1
	int location[9]={-1,-1,-1,-1,-1,-1,-1,-1,-1};//九个数字对应number中的下标
	int sortIndex[9];
	int revSort[9];

	cv::Rect temp_rect[9];//九宫格数字矩形
	cv::Point2f Position[9];//九宫格数字中心点的坐标

	int sudoku_top_position;
	int sudoku_bottom_position;
	int sudoku_left_position;
	int sudoku_right_position;

	tensorflow::Session *mSession;

	cv::Rect lastPosition;
//------------------------------------------------------------------------
	void update();

	int initSession(tensorflow::Session **pSession);//初始化模型

	int loadModel(tensorflow::Session *session, std::string modelPath);//加载模型

	void TensorOutputInit(tensorflow::Session* session,cv::Mat& img);

	/** 利用训练好的模型进行预测，获取图片为1-9数字的可能概率
	*
	* @param: Session *session
	* @param: Mat& img：需要检测的图片，大小为28x28
	* @param: double result[9]：存储数字1-9的概率
	* @return: void
	*/
	void GetTensorOutput(tensorflow::Session* session,cv::Mat& img,double result[9]);

	/** 图片识别与分割
	*
	* @param: Mat& Image：传入源图片的灰度图
	* @return: int：1失败，0成功
	*/
	int GetSudoku(cv::Mat& srcImage);

	void screen(cv::Mat& srcImage,std::vector<cv::Rect>& rects);//根据距离关系进行区域筛选

	void revision(int a,int b,int num);//预测修正
//----------------------------------------------------------------------------
public:
	int init(std::string filename,std::string modelPath);

	void clear();

	/** 大符九宫格分割识别主逻辑
	*
	* @param: Mat& srcImage：传入的源图片，即从相机获取的未处理的图片
	* @param: int top_position：顶部像素
	* @param: int left_position：左部像素
	* @param: int right_position：右部像素
	* @return: int：1失败，0成功
	*/
	int run(cv::Mat& srcImage,int top_position,int left_position,int right_position,int size);

	/** 大符九宫格排序
	*
	* @param: int num_Sudoku[9]：未排序的九宫格数字
	* @return: int：1失败，0成功
	*/
	void SudokuSort(int num_Sudoku[9]);//排序

	/** 获得所要击打的数字的位置
	*
	* @param: int n:model为0，n为所要击打的数字-1;model为1，n为排序后的第几个格子
	* @param: vector<Point2f>& points2d:该数字在图中的像素位置（最终所需输出）
 	* @param: vector<Point3f>& points3f:该数字的实际位置
	* @param: int model
	* @return: void
	*/
	void GetPosition(int n,std::vector<cv::Point2f>& points2d,std::vector<cv::Point3f>& points3f,int model);
//----------------------------------测试用-------------------------------------
	void DrawBox(int n,cv::Mat& srcImage);//显示检测框和命中框

	void SavePicture(int count);
};

#endif
