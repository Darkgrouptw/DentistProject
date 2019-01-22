#pragma once
#include <iostream>
#include <fstream>
#include <vector>

#include <QString>
#include <QFile>

struct RawDataProperty 
{
	// 詳情請見 TRCudaV2.cu 裡的 RawDataToPointCloud 裡的解說
	int SizeX;
	int SizeY;
	int SizeZ;
	long ShiftValue;
	double K_Step;
	int CutValue;
};

class DataManager
{
public:
	DataManager(void);
	~DataManager(void);

	//////////////////////////////////////////////////////////////////////////
	// 讀取校正檔
	//////////////////////////////////////////////////////////////////////////
	void ReadCalibrationData();

	// 參數
	RawDataProperty prop;							// 傳入的參數
	float* MappingMatrix;

private:
	// 這邊目前是不會用到
	// 後面要把所有的 Raw Data 的資訊 & 設定丟到這裡
	int Mapping_X;
	int Mapping_Y;
	float zRatio;

};

