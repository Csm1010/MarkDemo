////////////////////////////////////////////////////////////////////////////////
///Copyright(c)     UESTC ROBOMASTER2018      Model Code for Mark
///ALL RIGHTS RESERVED
///@file: mark.h
///@brief: 神符主逻辑头文件
///@vesion 2.9(版本号)
///@author: csm
///@email: 448554615@qq.com
///@date: 18-7-29
////////////////////////////////////////////////////////////////////////////////
#ifndef SMALL_MARK_H
#define SMALL_MARK_H

#include "small_mark_sudoku.h"
#include "big_mark_sudoku.h"
#include "mark_LED.h"
#include "rect_location.h"
#include "save_picture.h"

namespace mark
{
	class Mark
	{
	public:
		int init(std::string filename_mark,std::string smallmark_model_path,std::string bigmark_model_path);
		int clear();

		/*
		* return:
		0:成功
		1:LED分割or识别失败
		2:九宫格分割失败
		3:九宫格数字未变动
		4:缓冲时间段
		5:第二次运行过程中数字发生了变化
		6:其他
		*/
		int autoSmallMarkProcess(cv::Mat srcImage);//自动小符
		int autoBigMarkProcess(cv::Mat srcImage);//自动大符

		float getPitch();
		float getYaw();
		int getIdealExposure();
		int getSign();

		void setCurrentPitch(float p);
	
		void back();

	private:
		int IdeaExposure;
		float WIDTH_TARGET;
		float HEIGHT_TARGET;

		SmallMark_Sudoku* pSmallMark_Sudoku;
		Mark_LED* pMark_LED;
		RectLocation* pRectLocation;
		BigMark_Sudoku* pBigMark_Sudoku;
		SavePicture* pSavePicture;

		int num_LED_before[5];
		int num_Sudoku_before[9];
		int sign;//标志第几个LED数字
		int change;//标志在上一次击打之后九宫格数字有没有变化
		int buff;

		std::vector<cv::Point3f> points3d;

		float Pitch;
		float Yaw;

		float currentPitch;
//-------------------------------------------------
		int count;//标号,用于保存图片
	};
}

#endif
