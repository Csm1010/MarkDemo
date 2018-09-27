////////////////////////////////////////////////////////////////////////////////
///Copyright(c)     UESTC ROBOMASTER2018      Model Code for Mark
///ALL RIGHTS RESERVED
///@file: mark.cpp
///@brief: 神符主逻辑源文件
///@vesion 2.9(版本号)
///@author: csm
///@email: 448554615@qq.com
///@date: 18-7-29
////////////////////////////////////////////////////////////////////////////////
#include "mark.h"
#define DEBUG
//#define SAVE
//#define PAUSE
//#define OUTPUT

using namespace std;
using namespace cv;

namespace mark
{
	int Mark::init(string filename_mark,string smallmark_model_path,string bigmark_model_path)
	{
		FileStorage f(filename_mark, FileStorage::READ);
		f["EXPOSURE"] >> IdeaExposure;
		f["WIDTH_TARGET_SMALL_MARK"] >> WIDTH_TARGET;
		f["HEIGHT_TARGET_SMALL_MARK"] >> HEIGHT_TARGET;
		f.release();

		/*FileStorage fr("../res/recode.yaml", FileStorage::READ);
		fr["TIMES"] >> times;
		fr.release();

		if(times>100) times=0;

		FileStorage fw("../res/recode.yaml", FileStorage::WRITE);
		fw << "TIMES" << times+1;
		fw.release();*/

		pSmallMark_Sudoku = new SmallMark_Sudoku;
		pBigMark_Sudoku = new BigMark_Sudoku;
		pMark_LED = new Mark_LED;
		pRectLocation = new RectLocation;
		pSavePicture = new SavePicture;

		pSmallMark_Sudoku->init(filename_mark, smallmark_model_path);
		pBigMark_Sudoku->init(filename_mark,bigmark_model_path);
		pMark_LED->init(filename_mark);
		pRectLocation->init(filename_mark);
		pSavePicture->init();

		double half_x = WIDTH_TARGET / 2.0;
		double half_y = HEIGHT_TARGET / 2.0;
		points3d.emplace_back(Point3f(-half_x, -half_y, 0));
		points3d.emplace_back(Point3f(half_x, -half_y, 0));
		points3d.emplace_back(Point3f(half_x, half_y, 0));
		points3d.emplace_back(Point3f(-half_x, half_y, 0));

		count=0;
	}

	int Mark::clear() 
	{
		for (int i = 0; i < 9; ++i)
			num_Sudoku_before[i] = 0;
		for (int i = 0; i < 5; ++i)
			num_LED_before[i] = 0;
		sign = 0;
		change = 1;
		buff = -1;
		pMark_LED->clear();
		pSmallMark_Sudoku->clear();
		pBigMark_Sudoku->clear();
	}

	int Mark::autoSmallMarkProcess(Mat srcImage)
	{
		try
		{
			#ifdef PAUSE
			waitKey(0);
			#endif

			int flag = 0;
			int code;
			int num_Sudoku[9];
			int num_LED[5];
			Mat Image = srcImage.clone();

			#ifdef DEBUG
			namedWindow("src",WINDOW_NORMAL);
			imshow("src",srcImage);
			waitKey(1);
			#endif 

			#ifdef SAVE
			pSavePicture->save(srcImage);
			#endif

			code = pMark_LED->run(Image, num_LED);
			if (code == 1)
				return 1;

			int sudoku_top_position=pMark_LED->LED_bottom_position;
			int sudoku_left_position=pMark_LED->LED_left_position;
			int sudoku_right_position=pMark_LED->LED_right_position;
			int led_width=sudoku_right_position-sudoku_left_position;
			sudoku_right_position=sudoku_right_position+led_width/7*6;
			sudoku_left_position=sudoku_left_position-led_width/7*6;

			if(sudoku_top_position==srcImage.rows) return 1;
			if(sudoku_left_position<0) sudoku_left_position=0;
			if(sudoku_right_position>srcImage.cols) sudoku_right_position=srcImage.cols;

			code = pSmallMark_Sudoku->run(Image,sudoku_top_position,sudoku_left_position,sudoku_right_position,pMark_LED->LEDsize);

			if (code == 1)
				return 2;

			pSmallMark_Sudoku->SudokuSort(num_Sudoku);
			#ifdef SAVE
			//pSmallMark_Sudoku->SavePicture(count);
			//count++;
			#endif

			#ifdef OUTPUT
			for(int i = 0; i < 9; ++i)
				cout << num_Sudoku[i] << " ";
			cout << endl;
			#endif

			int diff_time = 0;
			if(num_Sudoku_before[0]!=0)
			{
				for (int i = 0; i < 9; ++i)
				{
					if (num_Sudoku[i] != num_Sudoku_before[i])
						++diff_time;
					if (diff_time>2) {change=1;break;}
					if (i == 8&&diff_time==0)
					{
						flag = 1;
						break;
					}
				}
			}
			else
				flag=1;
			for (int i = 0; i < 9; ++i)
				num_Sudoku_before[i] = num_Sudoku[i];

			#ifdef OUTPUT
			cout<<"flag:"<<flag<<" change:"<<change<<endl;		
			#endif			

			if (change ==0) {buff=0;return 3;}
			if (flag == 0) {buff=0;return 4;}

			++buff;
			if(buff<1) return 4;

			diff_time = 0;
			//cout<<sign<<endl;
			for (int i = 0; i < 5; ++i) 
			{
				if (num_LED[i] != num_LED_before[i])
					++diff_time;
				if (diff_time > 1)
				{
					sign = 0;
					break;
				}
				if (i == 4) ++sign;
			}
			for (int i = 0; i < 5; ++i)
				num_LED_before[i] = num_LED[i];

			sign = sign % 5;

			//if(sign>=5) break;
			//pSmallMark_Sudoku->ShowPicture();

			pSmallMark_Sudoku->DrawBox(num_LED[sign], srcImage);
			//cout << "num:" << num_LED[sign] << endl << endl;

			#ifdef DEBUG
			namedWindow("fin",WINDOW_NORMAL);
			imshow("fin",srcImage);
			waitKey(1);
			#endif

			vector<Point2f> points2d;
			pSmallMark_Sudoku->GetPosition(num_LED[sign], points2d,0);

			int box;
			for(int i=0;i<9;++i)
			{
				if(num_LED[sign]==num_Sudoku[i])
				{
					box=i;
					break;
				}
			}
			pRectLocation->GetPitchYaw(points2d, points3d,box, Pitch, Yaw);

			change = 0;
			buff = 0;

			cout<<"shoot number:"<<num_LED[sign]<<endl;
		}
		catch (...) 
		{
			pMark_LED->clear();
			pSmallMark_Sudoku->clear();
			return -1;
		}
		return 0;
	}

	int Mark::autoBigMarkProcess(Mat srcImage)
	{
		try
		{
			#ifdef PAUSE
			waitKey(0);
			#endif

			int flag = 0;
			int code;
			int num_Sudoku[9];
			int num_LED[5];
			Mat Image = srcImage.clone();

			#ifdef DEBUG
			namedWindow("src",WINDOW_NORMAL);
			imshow("src",srcImage);
			waitKey(1);
			#endif 
		
			#ifdef SAVE
			pSavePicture->save(srcImage);
			#endif

			code = pMark_LED->run(Image, num_LED);
			if (code == 1)
				return 1;

			int sudoku_top_position=pMark_LED->LED_bottom_position;
			int sudoku_left_position=pMark_LED->LED_left_position;
			int sudoku_right_position=pMark_LED->LED_right_position;
			int led_width=sudoku_right_position-sudoku_left_position;
			sudoku_right_position=sudoku_right_position+led_width/7*5;
			sudoku_left_position=sudoku_left_position-led_width/7*5;

			if(sudoku_top_position==srcImage.rows) return 1;
			if(sudoku_left_position<0) sudoku_left_position=0;
			if(sudoku_right_position>srcImage.cols) sudoku_right_position=srcImage.cols;

			code = pBigMark_Sudoku->run(Image,sudoku_top_position,sudoku_left_position,sudoku_right_position,pMark_LED->LEDsize);

			if (code == 1)
				return 2;

			pBigMark_Sudoku->SudokuSort(num_Sudoku);
			#ifdef SAVE
			//pBigMark_Sudoku->SavePicture(count);
			//count++;
			#endif

			#ifdef OUTPUT
			for(int i = 0; i < 9; ++i)
				cout << num_Sudoku[i] << " ";
			cout<<endl;
			#endif

			int diff_time = 0;
			if(num_Sudoku_before[0]!=0)
			{
				for (int i = 0; i < 9; ++i)
				{
					if (num_Sudoku[i] != num_Sudoku_before[i])
						++diff_time;
					if (diff_time > 2) {change=1;break;}
					if (i == 8&&diff_time == 0)
					{
						flag = 1;
						break;
					}
				}
			}
			else
				flag=1;
			for (int i = 0; i < 9; ++i)
				num_Sudoku_before[i] = num_Sudoku[i];

			#ifdef OUTPUT
			cout<<"flag:"<<flag<<" change:"<<change<<endl;
			#endif
			if (change==0) {buff=0; return 3;}
			if (flag==0) {buff=0;return 4;}
		
			++buff;
			if(buff<1) return 4;

			diff_time = 0;
			for (int i = 0; i < 5; ++i)
			{
				if (num_LED[i] != num_LED_before[i])
					++diff_time;
				if (diff_time > 1) 
				{
					sign = 0;
					break;
				}
				if (i == 4) ++sign;
			}
			for (int i = 0; i < 5; ++i)
				num_LED_before[i] = num_LED[i];

			sign = sign % 5;

			//if(sign>=5) break;

			pBigMark_Sudoku->DrawBox(num_LED[sign], srcImage);
			//cout << "num:" << num_LED[sign] << endl << endl;

			#ifdef DEBUG
			namedWindow("fin",WINDOW_NORMAL);
			imshow("fin",srcImage);
			waitKey(1);
			#endif

			vector<Point2f> points2d;
			vector<Point3f> points3d_;
			pBigMark_Sudoku->GetPosition(num_LED[sign], points2d, points3d_, 0);

			int box;
                        for(int i=0;i<9;++i)
                        {
                                if(num_LED[sign]==num_Sudoku[i])
                                {
                                        box=i;
                                        break;
                                }
                        }
			pRectLocation->GetPitchYaw(points2d, points3d_,box, Pitch, Yaw);

			change = 0;
			buff = 0;

			cout<<"shoot number:"<<num_LED[sign]<<endl;
		}
		catch(...)
		{
			pMark_LED->clear();
			pBigMark_Sudoku->clear();
			return -1;
		}
		return 0;
	}

	int Mark::getIdealExposure() 
	{
		return IdeaExposure;
	}

	float Mark::getPitch() 
	{
		return Pitch;
	}

	float Mark::getYaw() 
	{
		return Yaw;
	}

	int Mark::getSign()
	{
		return sign;
	}

	void Mark::setCurrentPitch(float p)
	{
		currentPitch=p;
	}

	void Mark::back()
	{
		if(sign>0) sign=(sign-1)%5;
	}
}
