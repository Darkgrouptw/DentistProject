#include "PointCloudInfo.h"

PointCloudInfo::PointCloudInfo()
{
}
PointCloudInfo::~PointCloudInfo()
{
}

void PointCloudInfo::ReadFromASC(QString FileName)
{
	// 開啟檔案
	QFile file(FileName);
	assert(file.open(QIODevice::ReadOnly));
	cout << "讀取點雲: " << FileName.toStdString() << endl;

	// 初始化變數
	float a, b, c;
	QTextStream ss(&file);
	Points.clear();

	// 點資訊
	while (!ss.atEnd())
	{
		ss >> a >> b >> c;

		QVector3D p(a, b, c);
		Points.push_back(p);
	}
	
	// 關閉檔案
	file.close();
	cout << "讀取完成!!" << endl;
}
void PointCloudInfo::SaveASC(QString FileName)
{
	// 開啟檔案
	QFile file(FileName);
	assert(file.open(QIODevice::WriteOnly));
	cout << "寫入檔案: " << FileName.toStdString() << endl;

	// 初始化變數
	QTextStream ss(&file);
	for (int i = 0; i < Points.size(); i++)
	{
		QVector3D pos = Points[i];
		ss << pos.x() << " " << pos.y() << " " << pos.z() << endl;
	}

	// 關閉檔案
	file.close();
	cout << "存檔完成!!" << endl;
}

// Helper Function
//void PointCloudInfo::FindCenterPoint()
//{
//	#pragma region 例外判斷
//	// 這邊是先做個判斷，裡面一定要有資料
//	assert(Points.size() > 0);
//	#pragma endregion
//	#pragma region 初始化變數
//	// 初始化變數
//	QVector3D BoundingBox_Max(-100, -100, -100);
//	QVector3D BoundingBox_Min(100, 100, 100);
//	
//	float tempX, tempY, tempZ;
//	#pragma endregion
//	#pragma region 跑每一個點去做比較
//	for (int i = 0; i < Points.size(); i++)
//	{
//		// 取點
//		tempX = Points[i].x();
//		tempY = Points[i].y();
//		tempZ = Points[i].z();
//	
//		// 找大於某一個點
//		if (BoundingBox_Max.x() < tempX)
//			BoundingBox_Max.setX(tempX);
//		if (BoundingBox_Max.y() < tempY)
//			BoundingBox_Max.setY(tempY);
//		if (BoundingBox_Max.z() < tempZ)
//			BoundingBox_Max.setZ(tempZ);
//
//		// 找小於某一個點
//		if (BoundingBox_Min.x() > tempX)
//			BoundingBox_Min.setX(tempX);
//		if (BoundingBox_Min.y() > tempY)
//			BoundingBox_Min.setY(tempY);
//		if (BoundingBox_Min.z() > tempZ)
//			BoundingBox_Min.setZ(tempZ);
//	}
//	#pragma endregion
//}