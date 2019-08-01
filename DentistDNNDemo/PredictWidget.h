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
	// 繪畫 Function
	//////////////////////////////////////////////////////////////////////////
	void initializeGL();
	void paintGL();

	//////////////////////////////////////////////////////////////////////////
	// 外部呼叫函式
	//////////////////////////////////////////////////////////////////////////
	void ProcessImg(Mat, int, float);

private:
	//////////////////////////////////////////////////////////////////////////
	// 繪圖相關
	//////////////////////////////////////////////////////////////////////////
	QOpenGLShaderProgram	*Program = NULL;
	GLuint					VertexBuffer = -1;
	GLuint					UVBuffer = -1;
	GLuint					DepthBuffer = -1;									// 這裡的深度，不是 Render 的深度，是牙齦到齒槽骨的深度
	QOpenGLTexture			*OtherSideTexture = NULL;

	//////////////////////////////////////////////////////////////////////////
	// Helper Function
	//////////////////////////////////////////////////////////////////////////
	QImage Mat2QImage(Mat const&, int);


	float Xvalue = 0;
	float Yvalue = 0;

};
