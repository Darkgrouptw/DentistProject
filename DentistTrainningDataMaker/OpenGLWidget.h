#pragma once
#include <iostream>
#include <cmath>

#include <QPoint>
#include <QVector2D>
#include <QVector3D>
#include <QVector4D>
#include <QMatrix4x4>
#include <QMouseEvent>
#include <QOpenGLWidget>
#include <QOpenGLFunctions_4_5_Core>

#include "OpenMesh/Core/IO/MeshIO.hh"
#include "OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh"

typedef OpenMesh::TriMesh_ArrayKernelT<> MeshType;

using namespace std;

class OpenGLWidget : public QOpenGLWidget, protected QOpenGLFunctions_4_5_Core
{
public:
	OpenGLWidget(QWidget*);
	~OpenGLWidget();

	void initializeGL();
	void paintGL();

	// 滑鼠事件
	void mousePressEvent(QMouseEvent *);
	void mouseMoveEvent(QMouseEvent *);

	// Connection Funciton
	bool LoadSTLFile(QString);

private:
	void CalcMatrix();						// 重算矩陣

	#pragma region 畫畫 Function
	void DrawGround();
	void DrawSTL();

	const float GridSize = 10;
	QVector2D GridMin = QVector2D(-GridSize, -GridSize);
	QVector2D GridMax = QVector2D(GridSize, GridSize);

	// MVP 矩陣
	QMatrix4x4		ProjectionMatrix;
	QMatrix4x4		ViewMatrix;

	const float		ElevationAngle = 30;	// 仰角
	const float		Radius = 30;			// 半徑
	int				ArcAngle = 0;			// 角度
	int				TempArcAngle = 0;		// 暫存角度 (For 滑鼠滑動使用)

	// Mesh
	MeshType		STLFile;
	QVector3D		BoundingBox[2];			// 最大的點 & 最小的點
	QMatrix4x4		TransformMatrix;		// 這邊是在做當 Load 進來的模型很大的時候，會做一個縮放的動作
	bool			IsLoaded = false;
	#pragma endregion
	#pragma region 拖移
	QPoint			PressPoint;
	QPoint			CurrentPoint;
	#pragma endregion
};

