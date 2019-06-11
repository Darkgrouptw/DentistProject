#include "PointCloudInfo.h"

PointCloudInfo::PointCloudInfo()
{
}
PointCloudInfo::~PointCloudInfo()
{
}

// 讀存檔
void PointCloudInfo::ReadFromXYZ(QString FileName)
{
	// 開啟檔案
	QFile file(FileName);
	assert(file.open(QIODevice::ReadOnly));
	cout << "讀取點雲: " << FileName.toLocal8Bit().toStdString() << endl;

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
	ReCalcCenterPos();		// 並算出中心點

	cout << "讀取完成!!" << endl;
}
void PointCloudInfo::SaveXYZ(QString FileName)
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

// 其他呼叫函式
void PointCloudInfo::ReCalcCenterPos()
{
	CenterPoint = QVector3D();
	for (int i = 0; i < Points.size(); i++)
		CenterPoint += Points[i];
	CenterPoint /= Points.size();
}
