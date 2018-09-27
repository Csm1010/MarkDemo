////////////////////////////////////////////////////////////////////////////////
///Copyright(c)     UESTC ROBOMASTER2018      Model Code for Mark
///ALL RIGHTS RESERVED
///@file: save_picture.cpp
///@brief: 保存图片源文件
///@vesion 2.9(版本号)
///@author: csm
///@email: 448554615@qq.com
///@date: 18-7-29
////////////////////////////////////////////////////////////////////////////////
#include "save_picture.h"
using namespace cv;
void SavePicture::run()
{
	while(true)
	{
		if(saveflag&&!mImgTmp.empty())
		{
			//保存操作,缓存区图片保存进入硬盘
			string name=format("..//res//picture//%d-%d.jpg",times,count);
			imwrite(name,mImgTmp);
			count++;
			saveflag=false;
			cout<<times<<"-"<<count<<" save!"<<endl;
		}
		else
			usleep(30000);
	}
}

void SavePicture::init()
{
    FileStorage fr("../res/recode.yaml", FileStorage::READ);
    fr["TIMES"] >> times;
    fr.release();

    if(times>100) times=0;

    FileStorage fw("../res/recode.yaml", FileStorage::WRITE);
    fw << "TIMES" << times+1;
    fw.release();

    saveflag=false;
    count=0;

    start();
}

int SavePicture::save(Mat image)
{
    if(!saveflag&&!image.empty()) {
        pthread_mutex_lock(&imgMutex);
        image.copyTo(mImgTmp);//写入mImgTmp,加锁
        pthread_mutex_unlock(&imgMutex);
        saveflag = true;
        return 0;
    }
    return -1;
}
