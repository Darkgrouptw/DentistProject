#pragma once
#include <iostream>
#include <cmath>

#include <QVector2D>
#include <QVector3D>
#include <QMatrix4x4>
#include <QOpenGLWidget>
#include <QOpenGLFunctions_4_5_Core>

#include "OpenMesh/Core/IO/MeshIO.hh"
#include "OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh"

typedef OpenMesh::TriMesh_ArrayKernelT<> Mesh;

using namespace std;

class OpenGLWidget : public QOpenGLWidget, protected QOpenGLFunctions_4_5_Core
{
public:
	OpenGLWidget(QWidget*);
	~OpenGLWidget();

	void initializeGL();
	void paintGL();


	// Connection Funciton
	bool LoadSTLFile(QString);

private:
	void CalcMatrix();						// 重算矩陣

	#pragma region 畫畫 Function
	void DrawGrid();

	QVector2D GridMin = QVector2D(-10, -10);
	QVector2D GridMax = QVector2D(10, 10);

	// MVP 矩陣
	QMatrix4x4 ProjectionMatrix;
	QMatrix4x4 ViewMatrix;

	const float		ElevationAngle = 30;	// 仰角
	const float		Radius = 15;			// 半徑
	int				ArcAngle = 0;			// 角度
	#pragma endregion

	Mesh STLFile;
};

