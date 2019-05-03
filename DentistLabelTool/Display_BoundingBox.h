#pragma once
#include <QtMath>
#include <QGLWidget>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QOpenGLWidget>
#include <QOpenGLShader>
#include <QOpenGLShaderProgram>
#include <QOpenGLFunctions_4_5_Core>

#include <iostream>
#include <vector>
using namespace std;

#include "Square.h"

class Display_BoundingBox : public QOpenGLWidget, protected QOpenGLFunctions_4_5_Core
{
public:
	Display_BoundingBox(QWidget*);
	~Display_BoundingBox();

	//////////////////////////////////////////////////////////////////////////
	// 繪製 Function
	//////////////////////////////////////////////////////////////////////////
	void initializeGL();
	void paintGL();

	//////////////////////////////////////////////////////////////////////////
	// 滑鼠事件
	//////////////////////////////////////////////////////////////////////////
	void		mousePressEvent(QMouseEvent *);
	void		mouseMoveEvent(QMouseEvent *);
	void		mouseReleaseEvent(QMouseEvent *);
	void		wheelEvent(QWheelEvent *);

	//////////////////////////////////////////////////////////////////////////
	// 外部連結
	//////////////////////////////////////////////////////////////////////////
	void LoadTexture(QImage, int);
	void SetFbo();

	Square* square;

private:
	//////////////////////////////////////////////////////////////////////////
	// OpenGL 相關
	//////////////////////////////////////////////////////////////////////////
	GLuint* texture;

	void		CalcMatrix();						// 重算矩陣

	//////////////////////////////////////////////////////////////////////////
	// MVP 矩陣
	//////////////////////////////////////////////////////////////////////////
	QMatrix4x4						ProjectionMatrix;
	QMatrix4x4						ViewMatrix;

	GLfloat ProjectionMatrixS[16];
	GLfloat ModelViewMatrixS[16];

#pragma region 滑鼠參數(沒用到)
	QPoint							PressPoint;						//拖曳
	QPoint							CurrentPoint;

	int								Radius = 15;					// 半徑
	const int						MaxRadius = 101;
	const int						MinRadius = 16;
	int								RadiusSpeed_Dev = 3;

	int								ElevationAngle = 30;			// 仰角
	int								TempElevationAngle = 0;

	int								ArcAngle = 0;					// 角度
	int								TempArcAngle = 0;				// 暫存角度 (For 滑鼠滑動使用)

#pragma endregion
	//////////////////////////////////////////////////////////////////////////
	// 畫畫 相關
	//////////////////////////////////////////////////////////////////////////
	float InitPainterSize = 1.0f;								//初始畫布


	vector<QVector2D>				TempAreaPoint;				//暫存繪圖資訊

	GLuint							_fbo, _fbo_C, _fbo_D;		//Fbo(存出使用?)

	GLdouble model[16], proj[16]; GLint view[4];				////座標轉換

	vector<vector<QPoint>>			saveTempPoint;				//保存繪圖資料

	GLfloat							scaleSize = 1.0f;			//Zoom in/out
	GLfloat							movex = 0.0f;				//往X translate
	GLfloat							movey = 0.0f;				//往Y translate

	GLfloat							nowx, nowy, nextx, nexty;	//暫存位置

	int TexWidth;//圖片大小資訊?
	int TexHeight;
	float WHProportion = 1.0f;									//長寬比
	
public:
	//////////////////////////////////////////////////////////////////////////
	// slider bar 相關
	//////////////////////////////////////////////////////////////////////////
	int sliderValue = 60;										//
	

	bool Imagelayer0 = false;
	bool Imagelayer1 = false;

	bool DrawTranslateChange = false;
	bool OpenPainter = false;
};

