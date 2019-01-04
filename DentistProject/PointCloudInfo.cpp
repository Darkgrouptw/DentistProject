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
