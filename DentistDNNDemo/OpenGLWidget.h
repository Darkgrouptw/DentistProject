#pragma once
#include "zhangsuen.h"

#include <iostream>
#include <vector>
#include <algorithm>

#include <QImage>
#include <QVector>
#include <QOpenGLWidget>
#include <QOpenGLShader>
#include <QOpenGLShaderProgram>
#include <QOpenGLFunctions_4_5_Core>
#include <QOpenGLTexture>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

// Bounding Box 的 DataStruct
struct BoundingBoxDataStruct
{
	vector<cv::Point> contoursRaw;				// 這個是原始的邊界
	vector<cv::Point> contoursPoly;				// 這個是對輪廓做多邊形擬合之後的邊界
	Rect boundingRect;							// 框框框起來
};

class OpenGLWidget : public QOpenGLWidget, protected QOpenGLFunctions_4_5_Core
{
public:
	OpenGLWidget(QWidget *);
	~OpenGLWidget();

	//////////////////////////////////////////////////////////////////////////
	// 繪畫 Function
	//////////////////////////////////////////////////////////////////////////
	void initializeGL();
	void paintGL();

	//////////////////////////////////////////////////////////////////////////
	// 外部呼叫函式
	//////////////////////////////////////////////////////////////////////////
	void ProcessImg(Mat, Mat, QVector<Mat>, QVector2D, QVector2D);

private:
	//////////////////////////////////////////////////////////////////////////
	// 繪圖相關
	//////////////////////////////////////////////////////////////////////////
	QOpenGLShaderProgram	*Program = NULL;
	GLuint					VertexBuffer = -1;
	GLuint					UVBuffer = -1;
	QOpenGLTexture			*OtherSideTexture = NULL;
	QOpenGLTexture			*ProbTexture = NULL;

	//////////////////////////////////////////////////////////////////////////
	//
	//////////////////////////////////////////////////////////////////////////
	QVector<int> MeatBounding;					// 牙肉位置(pixel)
	QVector<int> DiseaseBounding;				// 齒槽骨位置(pixel)

	//////////////////////////////////////////////////////////////////////////
	// Helper Function
	//////////////////////////////////////////////////////////////////////////
	QImage Mat2QImage(Mat const&, int);
	Mat GetBoundingBox(Mat, QVector2D&, QVector2D&);
	static bool SortByContourPointSize(BoundingBoxDataStruct&, BoundingBoxDataStruct&);
};

