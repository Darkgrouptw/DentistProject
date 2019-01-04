#pragma once
/*
�o��O�޲z�Ҧ��˸m�� class (�]�t �ŪޡBOCT)
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
	
	// ��k
	void ReadFromASC(QString);							// Ū�ɮ�
	void SaveASC(QString);								// �n�g�X�ɮ� (�Ĥ@�欰�E�b �ĤG��}�l���I�����)

	//////////////////////////////////////////////////////////////////////////
	// �x�s�����
	//////////////////////////////////////////////////////////////////////////
	QVector<QVector3D>	Points;							// �I (�o�䪺�I)
	QQuaternion			Gyro;							// ����������T
	QMatrix4x4			TransforMatrix;					// ���Ĥ@���I������ܯx�}
};