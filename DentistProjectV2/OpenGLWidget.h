#pragma once
#include <iostream>
#include <cmath>
#include <time.h>

#include <QMap>
#include <QPoint>
#include <QVector>
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
#include "OBJLoader.h"

typedef OpenMesh::TriMesh_ArrayKernelT<> MeshType;

using namespace std;

//////////////////////////////////////////////////////////////////////////
// GL Program 相關資訊
//////////////////////////////////////////////////////////////////////////
struct ProgramInfo 
{
	QOpenGLShaderProgram *program;				// 繪畫的 Program

	// 矩陣資訊
	int					ProjectionMLoc = -1;
	int					ViewMLoc = -1;
	int					ModelMLoc = -1;
};

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
	void		SetRotationMode(bool);

	//////////////////////////////////////////////////////////////////////////
	// 其他元件的 Function
	//////////////////////////////////////////////////////////////////////////
	void		SetRawDataManager(RawDataManager*);
	int			OCTViewType = 0;

	// void		SetFixMode(bool);
private:
	//////////////////////////////////////////////////////////////////////////
	// 初始化
	//////////////////////////////////////////////////////////////////////////
	void		InitProgram();						// 初始化 Program
	ProgramInfo	LinkProgram(QString);				// 連接相關
	void		CalcMatrix();						// 重算矩陣
	float		CalcArea(QVector<float>);			// 給三個邊長，算面積
	QVector3D	SamplePoint(QVector<QVector3D>);

	#pragma region 畫畫 Function
	void							DrawGround();
	void							DrawPointCloud();
	void							DrawResetRotation();
	//void							DrawVolumeData();
	//void							DrawAxis();
	#ifdef DEBUG_DRAW_AVERAGE_ERROR_PC
	void							DrawAverageDebugPC();
	#endif

	const float						GridSize = 10;
	QVector2D						GridMin = QVector2D(-GridSize, -GridSize);
	QVector2D						GridMax = QVector2D(GridSize, GridSize);

	//////////////////////////////////////////////////////////////////////////
	// Shader
	// 1. 畫地板的 Shader
	// 2. 九軸的 Shader
	// 3. 點的 Shader
	// 4. VolumeData 的 Shader
	//////////////////////////////////////////////////////////////////////////
	QVector<ProgramInfo>			ProgramList;

	//////////////////////////////////////////////////////////////////////////
	// 更新 Buffers
	//////////////////////////////////////////////////////////////////////////
	void							UpdatePC();
	//void							UpdateVolumeData();

	//////////////////////////////////////////////////////////////////////////
	// Render Data
	//////////////////////////////////////////////////////////////////////////
	QVector<QVector3D>				GroundPoints;								// 地板
	QVector<QVector2D>				GroundUVs;
	QMatrix4x4						GroundModelM;
	QVector<QVector3D>				GyroModelPoints;							// OCT
	QMatrix4x4						GyroTranslateM;

	//////////////////////////////////////////////////////////////////////////
	// Buffer
	//////////////////////////////////////////////////////////////////////////
	GLuint							GroundVertexBuffer = -1;
	GLuint							GroundUVBuffer = -1;
	GLuint							GyroModelVertexBuffer = -1;
	QVector<GLuint>					PointCloudVertexBufferList;
	QVector<GLuint>					VolumeDataVertexBufferList;
	QVector<GLuint>					VolumeDataPointTypeBufferList;

	//////////////////////////////////////////////////////////////////////////
	// MVP 矩陣
	//////////////////////////////////////////////////////////////////////////
	QMatrix4x4						ProjectionMatrix;
	QMatrix4x4						ViewMatrix;
	QMatrix4x4						OCTView_ModelMatrix[2];

	//////////////////////////////////////////////////////////////////////////
	// 其他 Location
	//////////////////////////////////////////////////////////////////////////
	int								PointSizeLoc;
	int								IsCrurrentPCLoc;

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
	QVector3D						BoundingBox[2];					// 最大的點 & 最小的點
	QMatrix4x4						TransformMatrix;				// 這邊是在做當 Load 進來的模型很大的時候，會做一個縮放的動作
	QVector3D						OffsetToCenter;					// 這邊是位移

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
