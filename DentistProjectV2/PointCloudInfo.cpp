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
	float a, b, c, d, e;
	QTextStream ss(&file);
	Points.clear();

	// 一開始是陀螺儀的資料
	ss >> a >> b >> c >> d;
	Gyro = QQuaternion(a, b, c, d);

	// 點資訊
	while (!ss.atEnd())
	{
		ss >> a >> b >> c >> d >> e;

		// 先確認這個一定要是 0
		//cout << a << b << c << endl;
		assert(d == 0 && e == 0);

		QVector3D p(a, b, c);
		Points.push_back(p);
	}
	
	// 關閉檔案
	file.close();
}
void PointCloudInfo::SaveASC(QString)
{

}

// Helper Function
void PointCloudInfo::FindCenterPoint()
{
	#pragma region 例外判斷
	// 這邊是先做個判斷，裡面一定要有資料
	assert(Points.size() > 0);
	#pragma endregion
	#pragma region 初始化變數
	// 初始化變數
	QVector3D BoundingBox_Max(-100, -100, -100);
	QVector3D BoundingBox_Min(100, 100, 100);
	
	float tempX, tempY, tempZ;
	#pragma endregion
	#pragma region 跑每一個點去做比較
	for (int i = 0; i < Points.size(); i++)
	{
		// 取點
		tempX = Points[i].x();
		tempY = Points[i].y();
		tempZ = Points[i].z();
	
		// 找大於某一個點
		if (BoundingBox_Max.x() < tempX)
			BoundingBox_Max.setX(tempX);
		if (BoundingBox_Max.y() < tempY)
			BoundingBox_Max.setY(tempY);
		if (BoundingBox_Max.z() < tempZ)
			BoundingBox_Max.setZ(tempZ);

		// 找小於某一個點
		if (BoundingBox_Min.x() > tempX)
			BoundingBox_Min.setX(tempX);
		if (BoundingBox_Min.y() > tempY)
			BoundingBox_Min.setY(tempY);
		if (BoundingBox_Min.z() > tempZ)
			BoundingBox_Min.setZ(tempZ);
	}
	#pragma endregion
	#pragma region MyRegion 

	#pragma endregion

}
