////////////////////////////////////////////////////////////////////////////////
///Copyright(c)     UESTC ROBOMASTER2018      Model Code for Mark
///ALL RIGHTS RESERVED
///@file: test.cpp
///@brief: 无
///@vesion 2.9(版本号)
///@author: csm
///@email: 448554615@qq.com
///@date: 18-7-29
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include "mark.h"

using namespace std;
using namespace cv;

int main()
{
	mark::Mark myMark;
	myMark.init("../res/mark_conf.yaml","../res/small_mark.pb","../res/big_mark.pb");

	Mat img;
	int flag=0;
	int ret;

	char model;
	cout<<"s:自动小符 b:自动大符"<<endl;
	cin>>model;
	myMark.clear();

	for(int i=0;i<50;++i)
	{
		string name=format("..//res//picture//small//49-%d.jpg",i);
		img=imread(name);

		if(!flag)
		{
			switch (model)
			{
				case 's':
					ret = myMark.autoSmallMarkProcess(img);
					break;
				case 'b':
					ret = myMark.autoBigMarkProcess(img);
					break;
				default:
					ret = -1;
					cout<<"model error!again"<<endl;
					cin>>model;
				break;
			}
		}
		else cout<<"image error!"<<endl;
	}

	return 0;
}
