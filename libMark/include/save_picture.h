////////////////////////////////////////////////////////////////////////////////
///Copyright(c)     UESTC ROBOMASTER2018      Model Code for Mark
///ALL RIGHTS RESERVED
///@file: save_picture.h
///@brief: 保存图片头文件
///@vesion 2.9(版本号)
///@author: csm
///@email: 448554615@qq.com
///@date: 18-7-29
////////////////////////////////////////////////////////////////////////////////

#ifndef INFANTRYDEMO_SAVE_PICTURE_H
#define INFANTRYDEMO_SAVE_PICTURE_H
#include "base_thread.h"
#include <opencv2/opencv.hpp>

class SavePicture:public BaseThread
{
private:
    cv::Mat mImgTmp;//解码后的图像
    pthread_mutex_t imgMutex = PTHREAD_MUTEX_INITIALIZER; //互斥变量锁

    bool saveflag;
    int times;
    int count;
public:
    void init();
    void run();
    int save(cv::Mat image);
};
#endif //INFANTRYDEMO_SAVE_PICTURE_H
