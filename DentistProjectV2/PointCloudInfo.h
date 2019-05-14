#pragma once
/*
這邊是管理所有裝置的 class (包含 藍芽、OCT)
*/
#include <iostream>
#include <cassert>

#include <QFile>
#include <QString>
#include <QTextStream>
#include <QIODevice>
#include <QVector>
#include <QMatrix4x4>

using namespace std;

class PointCloudInfo
{
public:
	PointCloudInfo();
	~PointCloudInfo();

	// 方法
	void ReadFromASC(QString);							// 讀檔案
	void SaveASC(QString);								// 要寫出檔案 (第一行為九軸 第二行開始為點雲資料)

	//////////////////////////////////////////////////////////////////////////
	// 儲存的資料
	//////////////////////////////////////////////////////////////////////////
	QVector<QVector3D>	Points;							// 點 (這邊的點)
	//QQuaternion			Gyro;							// 陀螺儀的資訊
	QVector3D			CenterPoint;					// 中心點，用於以後再做縫合的時候，可以根據與這個點的距離

private:
	//////////////////////////////////////////////////////////////////////////
	// Helper Function
	//////////////////////////////////////////////////////////////////////////
	//void				FindCenterPoint();				// 找中心點
};