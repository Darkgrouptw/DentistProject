#pragma once
/*
這邊
*/
#include "DataManager.h"
#include "TRCuda.cuh"

#include <QDataStream>
#include <QByteArray>

#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>


class RawDataManager
{
public:
	RawDataManager();
	~RawDataManager();

	//////////////////////////////////////////////////////////////////////////
	// 底下這邊的步驟可以二選一
	// 可以：
	// rawDataToPCSet	=> 讀檔，然後把資料存起來
	// (未執行)	=> 直接從 OCT 讀資料
	//////////////////////////////////////////////////////////////////////////
	void ReadRawDataFromFile(QString);

	void RawToPointCloud();
	void TranformToIMG();
private:
	DataManager		DManager;
	TRcuda			theTRcuda;


	QByteArray buffer;
};

