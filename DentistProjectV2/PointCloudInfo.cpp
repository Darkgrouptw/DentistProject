#include "PointCloudInfo.h"

PointCloudInfo::PointCloudInfo()
{
}
PointCloudInfo::~PointCloudInfo()
{
}

// 讀存檔
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
	cout << "寫入檔案: " << FileName.toLocal8Bit().toStdString() << endl;

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

// 拼接
void PointCloudInfo::RotateConstantAngle(int times)
{
	#pragma region 先算出 CenterPos
	QVector3D centerPos;
	for (int i = 0; i < Points.size(); i++)
		centerPos += Points[i];
	centerPos /= Points.size();
	#pragma endregion
	#pragma region 旋轉某個角度
	for (int i = 0; i < Points.size(); i++)
	{
		// 點雲
		QVector3D p = (Points[i] - centerPos);

		// 旋轉矩陣
		QMatrix4x4 rotationMatrix;
		rotationMatrix.rotate(-45, 1, 0, 0);
		for (int j = 0; j < times; j++)
			rotationMatrix.rotate(ROTATION_ANGLE, 0, 1, 0);
		p = (rotationMatrix * QVector4D(p, 1)).toVector3D();

		// 轉回原本的角度
		p += QVector3D(0, centerPos.y(), 0);
		Points[i] = p;
	}
	#pragma endregion
}