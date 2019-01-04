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
#include <QVector3D>
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
	QQuaternion			Gyro;							// 陀螺儀的資訊
	QMatrix4x4			TransforMatrix;					// 對於第一片點雲的轉至矩陣
};