#include "DentistDNNDemo.h"

DentistDNNDemo::DentistDNNDemo(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);

	#pragma region 事件連結
	// 主要功能
	connect(ui.slidingBar,			SIGNAL(valueChanged(int)),	this, SLOT(SliderValueChange(int)));

	// 測試相關
	connect(ui.TestRenderingBtn,	SIGNAL(clicked()),			this, SLOT(TestRenderFunctionEvent()));
	connect(ui.TestReadRawDataBtn,	SIGNAL(clicked()),			this, SLOT(PredictResultTesting()));
	
	#pragma endregion
	#pragma region TcpNetwork
	tcpSocket = new QTcpSocket(this);

	connect(tcpSocket, SIGNAL(connected()), this, SLOT(TcpConnected()));
	connect(tcpSocket, SIGNAL(disconnected()), this, SLOT(TcpDisConnected()));
	connect(tcpSocket, SIGNAL(readyRead()), this, SLOT(TcpreadyRead()));
	#pragma endregion
	#pragma region UI 初始化
	QImage img("./Images/ColorMap.png");
	ui.ColorMapLabel->setPixmap(QPixmap::fromImage(img));
	#pragma endregion
}

// 主要功能
void DentistDNNDemo::SliderValueChange(int)
{
	ui.DisplayPanel->GetSliderValue(ui.slidingBar->value());
	ui.ColorMapCurrentValue->setText(ui.DisplayPanel->GetColorMapValue(ui.slidingBar->value()));
	ui.DisplayPanel->update();
}

// 測試相關函式
void DentistDNNDemo::TestRenderFunctionEvent()
{
	#pragma region Test 路徑
	//QString TestFilePath = "E:/DentistData/2019.06.21-AfterSmooth/TOOTH bone 1/";
	//QString TestFilePath = "E:/DentistData/2019.06.21-AfterSmooth/TOOTH bone 2/";
	//QString TestFilePath = "E:/DentistData/2019.06.21-AfterSmooth/TOOTH bone 3.1/";
	//QString TestFilePath = "E:/DentistData/2019.06.21-AfterSmooth/TOOTH bone 3.2/";
	//QString TestFilePath = "E:/DentistData/2019.06.21-AfterSmooth/TOOTH bone 7.1/";
	//QString TestFilePath = "E:/DentistData/2019.06.21-AfterSmooth/TOOTH bone 7.2/";
	//QString TestFilePath = "E:/DentistData/2019.06.21-AfterSmooth/TOOTH bone 8.1/";
	QString TestFilePath = "E:/DentistData/2019.06.21-AfterSmooth/TOOTH bone 9.1/";
	QString OtherSidePath_Predict = TestFilePath + "Predict.png";
	QString OtherSidePath_Org = TestFilePath + "OtherSide.png";
	QString BoundingBoxPath = TestFilePath + "boundingBox.txt";
	#pragma endregion
	#pragma region 讀 Bounding Box
	ReadBounding(BoundingBoxPath);
	#pragma endregion	
	#pragma region 讀圖
	Mat otherSideMat_Org		= imread(OtherSidePath_Org.toLocal8Bit().toStdString(), IMREAD_GRAYSCALE);
	Mat otherSideMat_Predict	= imread(OtherSidePath_Predict.toLocal8Bit().toStdString(), IMREAD_GRAYSCALE);
	QVector<Mat> FullMat;

	for (int i = 0; i <= 140; i++)
	{
		Mat mat = imread((TestFilePath + "Smooth/" + QString::number(i) + ".png").toLocal8Bit().toStdString(), IMREAD_COLOR);
		FullMat.push_back(mat);
	}
	((OpenGLWidget*)(ui.DisplayPanel))->ProcessImg(otherSideMat_Org, otherSideMat_Predict, FullMat, OrginTL, OrginBR, ui.ColorMapMaxValue, ui.ColorMapMinValue);
	#pragma endregion
	#pragma region 刷新
	ui.DisplayPanel->GetSliderValue(ui.slidingBar->value());
	ui.ColorMapCurrentValue->setText(ui.DisplayPanel->GetColorMapValue(ui.slidingBar->value()));
	ui.DisplayPanel->update();
	#pragma endregion
}
void DentistDNNDemo::ReadBounding(QString FileName) {
	#pragma region 讀取BoundingBox
	QFile file(FileName);
	file.open(QIODevice::ReadOnly);
	cout << "讀取boundingbox: " << FileName.toLocal8Bit().toStdString() << endl;

	// 初始化變數
	float a, b, c;
	QTextStream ss(&file);

	QVector<QVector2D> TL;
	QVector<QVector2D> BR;

	QString TempFile;

	float w, x, y, z;
	// 第一行不要
	TempFile = ss.readLine();

	while (!ss.atEnd())
	{
		// 讀一條
		TempFile = ss.readLine();
		if (TempFile == "")
			break;

		//xyz
		assert(TempStr.size == 3 && "讀取的資料有誤!!");
		w = TempFile.section(' ', 0, 0).trimmed().toFloat();
		x = TempFile.section(' ', 1, 1).trimmed().toFloat();
		y = TempFile.section(' ', 2, 2).trimmed().toFloat();
		z = TempFile.section(' ', 3, 3).trimmed().toFloat();

		//cout << w << " " << x << " " << y << " " << z << endl;

		TL.push_back(QVector2D(w, x));
		BR.push_back(QVector2D(y, z));
	}

	// 關閉檔案
	file.close();

	cout << "讀取BoundingBox完成!!" << endl;
	#pragma endregion
	#pragma region 得到最大boundbox range
	for (int i = 60 - 1; i <= 200 - 1; i++)
	{
		if (TL[i].x() < OrginTL.x())OrginTL.setX(TL[i].x());
		if (TL[i].y() < OrginTL.y())OrginTL.setY(TL[i].y());
		if (BR[i].x() > OrginBR.x())OrginBR.setX(BR[i].x());
		if (BR[i].y() > OrginBR.y())OrginBR.setY(BR[i].y());
	}
	#pragma endregion
}

void DentistDNNDemo::PredictResultTesting() {

	// 1. 先讀 Data
	ReadRawDataForBorderTest();

	// 2. 接著要抓出
	rawManager.NetworkDataGenerateInRamV2();
	if (!rawManager.CheckIsValidData())
	{
		cout << "Eigen 算有錯誤!!" << endl;
		return;
	}

	// 3. 存出所有Network需要圖片
	rawManager.SaveNetworkImage();

	// 4. 傳上伺服器Predict
	TcpNetwork();
}
void DentistDNNDemo::TcpConnected()
{
	qDebug() << "socket connected";

	QString string = Requestmsg;
	tcpSocket->write(string.toLatin1());
	qDebug() << "Send: " << string;
}
void DentistDNNDemo::TcpDisConnected()
{
	qDebug() << "socket disconnected";

	if (Requestmsg == "Sent") {
		Sefile.close();
		Requestmsg = "Wait";
		tcpSocket->connectToHost("140.118.175.94", 10000);
	}
	else if (Requestmsg == "Wait") {
		Requestmsg = "Recv";
		isStart = true;
		tcpSocket->connectToHost("140.118.175.94", 10000);
	}
	else if (Requestmsg == "Recv") {
		isStart = false;
		Sefile.close();
		useUNRAR();
		// 5. 把預測資料貼回原圖
		rawManager.LoadPredictImage();

		// 6. Smooth 結果並把點區塊刪除
		rawManager.SmoothNetworkData();

		// 7. 轉到 QImage 中
		//rawManager.NetworkDataToQImage();

		// 8. 顯示結果
		//rawManager.ShowImageIndex(60);
	}
}
void DentistDNNDemo::TcpreadyRead()
{
	qDebug() << "client readyRead";

	QByteArray buf = tcpSocket->readAll();
	QString string = QString::fromUtf8(buf.data());
	//qDebug() << "Read: " << buf;

	if (string == "StartSent") {
		useRAR();
		SentTest();
		tcpSocket->disconnectFromHost();
	}
	else if (string == "WorkDone") {
		cout << "PyOK" << endl;
		tcpSocket->disconnectFromHost();
	}
	else {
		RecvTest(buf);
	}
}
void DentistDNNDemo::TcpNetwork()
{
	Requestmsg = "Sent";
	tcpSocket->connectToHost("140.118.175.94", 10000);
}
void DentistDNNDemo::SentTest()
{
	//QString filePath = QFileDialog::getOpenFileName(this, "open", "./");
	QString filePath = "./Image.zip";
	if (filePath.isEmpty() == false) {
		fileName.clear();
		fileSize = 0;

		// 獲取文件訊息 : 名字、大小
		QFileInfo info(filePath);
		fileName = info.fileName();
		fileSize = info.size();
		sendSize = 0;
		cout << fileName.toStdString() << " " << fileSize << endl;

		// 讀檔
		Sefile.setFileName(filePath);
		if (Sefile.open(QIODevice::ReadOnly) == false) {
			cout << "File Not Open !" << endl;
		}
		else {
			QString head = QString("%1##%2").arg(fileName).arg(fileSize);
			qint64 len = tcpSocket->write(head.toUtf8());

			if (len < 0) {
				cout << "head Fail" << endl;
				Sefile.close();
			}
			qint64 slen = 0;
			do {
				char buf[BUF_SIZE] = { 0 };
				slen = 0;
				slen = Sefile.read(buf, BUF_SIZE);
				slen = tcpSocket->write(buf, slen);
				sendSize += slen;
			} while (slen > 0);
		}
	}
}
void DentistDNNDemo::RecvTest(QByteArray buf)
{
	if (isStart == true) {
		isStart = false;

		QString filePath = "./received_file.zip";

		if (filePath.isEmpty() == false) {
			fileName.clear();
			fileSize = 0;
			QFileInfo info(filePath);
			fileName = info.fileName();
			Sefile.setFileName(fileName);
		}
		if (Sefile.open(QIODevice::WriteOnly) == false) {
			cout << "File Not Open !" << endl;
		}
		Sefile.write(buf);
	}
	else {
		Sefile.write(buf);
	}
}
void DentistDNNDemo::useRAR()
{
	std::string AAA = "WinRAR.exe a -afzip Image.zip ./Predicts/*";

	system(AAA.c_str());
}
void DentistDNNDemo::useUNRAR()
{
	std::string AAA = "WinRAR.exe -o+ x received_file.zip";

	system(AAA.c_str());
}

void DentistDNNDemo::ReadRawDataForBorderTest()
{
	QString RawFileName = QFileDialog::getOpenFileName(this, codec->toUnicode("邊界測試"), "E:/DentistData/ScanData/", "", nullptr, QFileDialog::DontUseNativeDialog);
	if (RawFileName != "")
	{
		RawDataType type = rawManager.ReadRawDataFromFileV2(RawFileName);
		rawManager.TransformToIMG(false);

		if (type == RawDataType::MULTI_DATA_TYPE)
		{
			rawManager.TransformToOtherSideView();
		}
		else {
			cout << "資料不是MULTI" << endl;
			return;
		}
	}
}