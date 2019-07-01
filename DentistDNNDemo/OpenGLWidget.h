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
	float GetDistanceValue(int);
	void GetSliderValue(float);
	QString GetColorMapValue(int);
	int GetNowSliderValue(int);

private:
	//////////////////////////////////////////////////////////////////////////
	// 繪圖相關
	//////////////////////////////////////////////////////////////////////////
	QOpenGLShaderProgram	*Program = NULL;
	GLuint					VertexBuffer = -1;
	GLuint					UVBuffer = -1;
	GLuint					DepthBuffer = -1;									// 這裡的深度，不是 Render 的深度，是牙齦到齒槽骨的深度
	QOpenGLTexture			*OtherSideTexture = NULL;
	QOpenGLTexture			*ProbTexture = NULL;
	QOpenGLTexture			*DepthTexture = NULL;
	float					SliderValue = ((60.0f / 250.0f) - 0.5f) * 2.0f;		// Slider現在的位置(-1 ~ 1)
	void					DrawSlider();										// 畫Slider
	bool					CheckIsNonZeroValue = false;

	//////////////////////////////////////////////////////////////////////////
	// 校正到世界座標所需要的東西
	//////////////////////////////////////////////////////////////////////////
	QVector<int>			nonZeroIndex;										
	QVector<int>			MeatBounding;										// 牙肉位置(pixel)
	QVector<int>			BoneBounding;										// 齒槽骨位置(pixel)
	QVector<QVector2D>		WorldPosMeat;										// 世界座標
	QVector<QVector2D>		WorldPosBone;										// 同上
	QVector<float>			DistanceBounding;									// 算一下他們的距離
	float					DistanceMin, DistanceMax;							// 最大、最小值
	CalibrationUtility		calibrationTool;


	//////////////////////////////////////////////////////////////////////////
	// Helper Function
	//////////////////////////////////////////////////////////////////////////
	QImage Mat2QImage(Mat const&, int);
	Mat GetBoundingBox(Mat, QVector2D&, QVector2D&);
	static bool SortByContourPointSize(BoundingBoxDataStruct&, BoundingBoxDataStruct&);
	static bool	CompareContourArea(vector<cv::Point>, vector<cv::Point>);		// OpenCV
};

