﻿#pragma once
#include <iostream>
#include <cmath>
#include <time.h>

#include <QPoint>
#include <QVector2D>
#include <QVector3D>
#include <QVector4D>
#include <QMatrix4x4>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QOpenGLWidget>
#include <QOpenGLShader>
#include <QOpenGLShaderProgram>
#include <QOpenGLFunctions_4_5_Core>

#include "OpenMesh/Core/IO/MeshIO.hh"
#include "OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh"

#include "RawDataManager.h"

typedef OpenMesh::TriMesh_ArrayKernelT<> MeshType;

using namespace std;

class OpenGLWidget : public QOpenGLWidget, protected QOpenGLFunctions_4_5_Core
{
public:
	OpenGLWidget(QWidget*);
	~OpenGLWidget();

	void		initializeGL();
	void		paintGL();

	//////////////////////////////////////////////////////////////////////////
	// 滑鼠事件
	//////////////////////////////////////////////////////////////////////////
	void		mousePressEvent(QMouseEvent *);
	void		mouseMoveEvent(QMouseEvent *);
	void		wheelEvent(QWheelEvent *);

	//////////////////////////////////////////////////////////////////////////
	// Connection Funciton
	//////////////////////////////////////////////////////////////////////////
	bool		LoadSTLFile(QString);
	void		SetRenderTriangleBool(bool);
	void		SetRenderBorderBool(bool);
	void		SetRenderPointCloudBool(bool);
	void		SetRotationMode(bool);

	//////////////////////////////////////////////////////////////////////////
	// 其他元件的 Function
	//////////////////////////////////////////////////////////////////////////
	void		SetRawDataManager(RawDataManager*);

private:
	//////////////////////////////////////////////////////////////////////////
	// 初始化
	//////////////////////////////////////////////////////////////////////////
	void		InitProgram();						// 初始化 Program

	void		CalcMatrix();						// 重算矩陣
	float		CalcArea(QVector<float>);			// 給三個邊長，算面積
	QVector3D	SamplePoint(QVector<QVector3D>);

	#pragma region 畫畫 Function
	void							DrawGround();
	void							DrawPointCloud();
	void							DrawSTL();
	void							DrawResetRotation();

	bool							RenderTriangle_bool = true;
	bool							RenderBorder_bool = true;
	bool							RenderPointCloud_bool = false;

	const float						GridSize = 10;
	QVector2D						GridMin = QVector2D(-GridSize, -GridSize);
	QVector2D						GridMax = QVector2D(GridSize, GridSize);

	//////////////////////////////////////////////////////////////////////////
	// Shader
	//////////////////////////////////////////////////////////////////////////
	QOpenGLShaderProgram			*program = NULL;
	int								ProjectionMatrixLoc;
	int								ViewMatrixLoc;
	int								ModelMatrixLoc;

	//////////////////////////////////////////////////////////////////////////
	// Render Data
	//////////////////////////////////////////////////////////////////////////
	QVector<QVector3D>				VertexData;
	QVector<QVector3D>				BaryCentricData;

	//////////////////////////////////////////////////////////////////////////
	// Buffer
	//////////////////////////////////////////////////////////////////////////
	GLuint							VertexBuffer = -1;
	GLuint							BaryCentricBuffer = -1;

	//////////////////////////////////////////////////////////////////////////
	// MVP 矩陣
	//////////////////////////////////////////////////////////////////////////
	QMatrix4x4						ProjectionMatrix;
	QMatrix4x4						ViewMatrix;

	//////////////////////////////////////////////////////////////////////////
	// 其他設定
	//////////////////////////////////////////////////////////////////////////
	int								Radius = 30;					// 半徑
	const int						MaxRadius = 51;
	const int						MinRadius = 21;
	int								RadiusSpeed_Dev = 3;

	int								ElevationAngle = 30;			// 仰角
	int								TempElevationAngle = 0;

	int								ArcAngle = 0;					// 角度
	int								TempArcAngle = 0;				// 暫存角度 (For 滑鼠滑動使用)

	//////////////////////////////////////////////////////////////////////////
	// Mesh
	//////////////////////////////////////////////////////////////////////////
	OpenMesh::FPropHandleT<float>	AreaInfo;
	MeshType						STLFile;
	QVector<QVector3D>				PointArray;
	int								SpreadingPointSize = 10000;
	QVector3D						BoundingBox[2];					// 最大的點 & 最小的點
	QMatrix4x4						TransformMatrix;				// 這邊是在做當 Load 進來的模型很大的時候，會做一個縮放的動作
	QVector3D						OffsetToCenter;					// 這邊是位移

	bool							IsLoaded = false;

	//////////////////////////////////////////////////////////////////////////
	// Reset Rotation Mode
	//////////////////////////////////////////////////////////////////////////
	bool							RotationMode = false;

	//////////////////////////////////////////////////////////////////////////
	// 這邊是點雲資訊
	//////////////////////////////////////////////////////////////////////////
	RawDataManager*					rawManager = NULL;

	#pragma endregion
	#pragma region 拖移
	QPoint							PressPoint;
	QPoint							CurrentPoint;
	#pragma endregion
};