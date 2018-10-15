#pragma once
/*
這邊
*/
#include "DataManager.h"
#include "TRCuda.cuh"

#include <QVector>
#include <QDataStream>
#include <QByteArray>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


// 這邊是為了要讓邊界 Smooth 一點
struct IndexMapInfo
{
	int index;			// 位置資訊
	int ZValue;			// Z 的深度值是多少
};

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

	int				LerpFunction(int, int, int, int, int);

	QByteArray buffer;
};

