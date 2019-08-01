#pragma once
#include <QtWidgets/QMainWindow>
#include "ui_DentistDNNDemo.h"

#include <iostream>

#include <QVector>
#include <QVector2D>
#include <QFile>
#include <QImage>
#include <QPixmap>
#include <QFileDialog>
#include <QTimer>
#include <QDir>
#include <QDate>
#include <QTime>
#include <QMessageBox>
#include <QAbstractSocket>
#include <QTcpSocket>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "GlobalDefine.h"
#include "RawDataManager.h"

#define BUF_SIZE 1024*4				//TCP 傳送/接收資料的buffsize

using namespace std;

class DentistDNNDemo : public QMainWindow
{
	Q_OBJECT

public:
	DentistDNNDemo(QWidget *parent = Q_NULLPTR);

private:
	Ui::DentistDNNDemoClass ui;

	//////////////////////////////////////////////////////////////////////////
	// 其他變數 or 元件
	//////////////////////////////////////////////////////////////////////////
	RawDataManager	rawManager;													// 所有跟裝置有關的 (藍芽、OCT)
	QTextCodec *codec = QTextCodec::codecForName("Big5-ETen");

	//////////////////////////////////////////////////////////////////////////
	// TCP 相關變數 / 函式
	//////////////////////////////////////////////////////////////////////////
	QTcpSocket *tcpSocket;

	QFile Sefile;
	QString fileName;
	qint64 fileSize;
	qint64 sendSize;
	qint64 recvSize;

	QString Requestmsg;

	bool isStart = false; // 接收資料時的Lock
	void TcpNetwork();
	void useRAR();
	void useUNRAR();
	void SentTest();
	void RecvTest(QByteArray);

	// Read BoundingBoxFunc
	void ReadBounding(QString);
	bool showvalueLabel = false;

	// Bounding
	QVector2D OrginTL = QVector2D(9999, 9999);
	QVector2D OrginBR = QVector2D(-1, -1);

	QVector<Mat> FullMat;


private slots:
	//////////////////////////////////////////////////////////////////////////
	// 主要功能
	//////////////////////////////////////////////////////////////////////////
	void SliderValueChange(int);

	//////////////////////////////////////////////////////////////////////////
	// 測試相關 Function
	//////////////////////////////////////////////////////////////////////////
	void TestRenderFunctionEvent();
	void PredictResultTesting();
	void ReadRawDataForBorderTest();
	void TestValidDataEvent();

	//////////////////////////////////////////////////////////////////////////
	// TCP Function
	//////////////////////////////////////////////////////////////////////////
	void TcpConnected();
	void TcpDisConnected();
	void TcpreadyRead();

	//////////////////////////////////////////////////////////////////////////
	// Other
	//////////////////////////////////////////////////////////////////////////
	void ShowValue();
};
