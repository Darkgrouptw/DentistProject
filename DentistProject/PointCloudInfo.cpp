#include "PointCloudInfo.h"

PointCloudInfo::PointCloudInfo()
{
}
PointCloudInfo::~PointCloudInfo()
{
}

void PointCloudInfo::ReadFromASC(QString FileName)
{
	// �}���ɮ�
	QFile file(FileName);
	assert(file.open(QIODevice::ReadOnly));
	cout << "Ū���I��: " << FileName.toStdString() << endl;

	// ��l���ܼ�
	float a, b, c, d, e;
	QTextStream ss(&file);
	Points.clear();

	// �@�}�l�O�����������
	ss >> a >> b >> c >> d;
	Gyro = QQuaternion(a, b, c, d);

	while (!ss.atEnd())
	{
		ss >> a >> b >> c >> d >> e;

		// ���T�{�o�Ӥ@�w�n�O 0
		//cout << a << b << c << endl;
		assert(d == 0 && e == 0);

		QVector3D p(a, b, c);
		Points.push_back(p);
	}
	
	// �����ɮ�
	file.close();
}
void PointCloudInfo::SaveASC(QString)
{

}
