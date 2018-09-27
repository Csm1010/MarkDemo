////////////////////////////////////////////////////////////////////////////////
///Copyright(c)     UESTC ROBOMASTER2018      Model Code for Mark
///ALL RIGHTS RESERVED
///@file: small_mark_sudoku.h
///@brief: 小神符九宫格分割识别源文件
///@vesion 2.9(版本号)
///@author: csm
///@email: 448554615@qq.com
///@date: 18-7-29
////////////////////////////////////////////////////////////////////////////////
#ifndef SMALL_MARK_SUDOKU_H
#define SMALL_MARK_SUDOKU_H

#include <iostream>
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "opencv2/opencv.hpp"

class SmallMark_Sudoku
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

	int THRESH2;

	int LEDsize;
//---------------------------------------------------------------------------
	cv::Mat Sudoku[9];//分离出的九宫格图片
	double prob[9][9];//九宫格通过模型预测出的每个格子为1-9的概率
	int number[9] = {0};//九宫格的预测数字-1
	int location[9] = {-1, -1, -1, -1, -1, -1, -1, -1, -1};//九个数字对应number中的下标
	int sortIndex[9];

	cv::RotatedRect Position[9];//九宫格的位置

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
	void GetTensorOutput(cv::Mat &img, double result[9]);

	/** 图片识别与分割
	*
	* @param: Mat& Image：传入源图片的灰度图
	* @return: int：1失败，0成功
	*/
	int GetSudoku(cv::Mat &srcImage);

	int screen(cv::Mat &srcImage, std::vector <cv::RotatedRect> &rotaRects, std::vector <std::vector<cv::Point>> &contours,std::vector<int> &sign);//根据距离关系进行区域筛选

	void revision(int a, int b, int num);//预测修正
//----------------------------------------------------------------------------
public:
	int init(std::string filename, std::string modelPath);

	void clear();

	/** 小符九宫格运行主逻辑
	*
	* @param: Mat& srcImage：传入的源图片，即从相机获取的未处理的图片
	* @param: int top_position：顶部像素
	* @param: int left_position：左部像素
	* @param: int right_position：右部像素
	* @return: int：1失败，0成功
	*/
	int run(cv::Mat &srcImage,int top_position,int left_position,int right_position,int size);

	void SudokuSort(int num_sudoku[9]);//九宫格排序

	/** 获得所要击打的数字的位置
	*
	* @param: int n: model为0，n为所要击打的数字-1;model为1，n为排序后的第几个格子
	* @param: vector<Point2f>& points2d:该数字在图中的像素位置（最终输出）
	* @param: int model
	* @return: void
	*/
	void GetPosition(int n, std::vector <cv::Point2f> &points2d,int model);
//----------------------------------测试用-------------------------------------
	void DrawBox(int n, cv::Mat &srcImage);//显示检测框和命中框

	void SavePicture(int count);
};

#endif 
