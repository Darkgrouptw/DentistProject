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
	// ø�s Function
	//////////////////////////////////////////////////////////////////////////
	void initializeGL();
	void paintGL();

	//////////////////////////////////////////////////////////////////////////
	// �ƹ��ƥ�
	//////////////////////////////////////////////////////////////////////////
	void		mousePressEvent(QMouseEvent *);
	void		mouseMoveEvent(QMouseEvent *);
	void		mouseReleaseEvent(QMouseEvent *);
	void		wheelEvent(QWheelEvent *);

	//////////////////////////////////////////////////////////////////////////
	// �~���s��
	//////////////////////////////////////////////////////////////////////////
	void LoadTexture(QImage, int);
	void SetFbo();

	Square* square;

private:
	//////////////////////////////////////////////////////////////////////////
	// OpenGL ����
	//////////////////////////////////////////////////////////////////////////
	GLuint* texture;

	void		CalcMatrix();						// ����x�}

	//////////////////////////////////////////////////////////////////////////
	// MVP �x�}
	//////////////////////////////////////////////////////////////////////////
	QMatrix4x4						ProjectionMatrix;
	QMatrix4x4						ViewMatrix;

	GLfloat ProjectionMatrixS[16];
	GLfloat ModelViewMatrixS[16];

#pragma region �ƹ��Ѽ�(�S�Ψ�)
	QPoint							PressPoint;						//�즲
	QPoint							CurrentPoint;

	int								Radius = 15;					// �b�|
	const int						MaxRadius = 101;
	const int						MinRadius = 16;
	int								RadiusSpeed_Dev = 3;

	int								ElevationAngle = 30;			// ����
	int								TempElevationAngle = 0;

	int								ArcAngle = 0;					// ����
	int								TempArcAngle = 0;				// �Ȧs���� (For �ƹ��ưʨϥ�)

#pragma endregion
	//////////////////////////////////////////////////////////////////////////
	// �e�e ����
	//////////////////////////////////////////////////////////////////////////
	float InitPainterSize = 1.0f;								//��l�e��


	vector<QVector2D>				TempAreaPoint;				//�Ȧsø�ϸ�T

	GLuint							_fbo, _fbo_C, _fbo_D;		//Fbo(�s�X�ϥ�?)

	GLdouble model[16], proj[16]; GLint view[4];				////�y���ഫ

	vector<vector<QPoint>>			saveTempPoint;				//�O�sø�ϸ��

	GLfloat							scaleSize = 1.0f;			//Zoom in/out
	GLfloat							movex = 0.0f;				//��X translate
	GLfloat							movey = 0.0f;				//��Y translate

	GLfloat							nowx, nowy, nextx, nexty;	//�Ȧs��m

	int TexWidth;//�Ϥ��j�p��T?
	int TexHeight;
	float WHProportion = 1.0f;									//���e��
	
public:
	//////////////////////////////////////////////////////////////////////////
	// slider bar ����
	//////////////////////////////////////////////////////////////////////////
	int sliderValue = 60;										//
	

	bool Imagelayer0 = false;
	bool Imagelayer1 = false;

	bool DrawTranslateChange = false;
	bool OpenPainter = false;
};

