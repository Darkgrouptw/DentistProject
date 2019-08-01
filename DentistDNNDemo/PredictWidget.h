#pragma once
#include "zhangsuen.h"
#include "CalibrationUtility.h"

#include <iostream>
#include <vector>
#include <algorithm>

#include <QFile>
#include <QImage>
#include <QVector>
#include <QOpenGLWidget>
#include <QOpenGLShader>
#include <QOpenGLShaderProgram>
#include <QOpenGLFunctions_4_5_Core>
#include <QOpenGLTexture>
#include <QLabel>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

class PredictWidget : public QOpenGLWidget, protected QOpenGLFunctions_4_5_Core
{
public:
	PredictWidget(QWidget *);
	~PredictWidget();

	//////////////////////////////////////////////////////////////////////////
	// ø�e Function
	//////////////////////////////////////////////////////////////////////////
	void initializeGL();
	void paintGL();

	//////////////////////////////////////////////////////////////////////////
	// �~���I�s�禡
	//////////////////////////////////////////////////////////////////////////
	void ProcessImg(Mat, int, float);

private:
	//////////////////////////////////////////////////////////////////////////
	// ø�Ϭ���
	//////////////////////////////////////////////////////////////////////////
	QOpenGLShaderProgram	*Program = NULL;
	GLuint					VertexBuffer = -1;
	GLuint					UVBuffer = -1;
	GLuint					DepthBuffer = -1;									// �o�̪��`�סA���O Render ���`�סA�O���i�쾦�Ѱ����`��
	QOpenGLTexture			*OtherSideTexture = NULL;

	//////////////////////////////////////////////////////////////////////////
	// Helper Function
	//////////////////////////////////////////////////////////////////////////
	QImage Mat2QImage(Mat const&, int);


	float Xvalue = 0;
	float Yvalue = 0;

};
