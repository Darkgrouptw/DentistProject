#include "DentistProjectV2.h"

DentistProjectV2::DentistProjectV2(QWidget *parent) : QMainWindow(parent)
{
	ui.setupUi(this);
	#pragma region UpdateGLTimer
	UpdateGLTimer = new QTimer();
	#pragma endregion
	#pragma region 事件連結
	// OCT 相關(主要)
	connect(ui.SaveLocationBtn,								SIGNAL(clicked()),				this,	SLOT(ChooseSaveLocaton()));
	connect(ui.AutoSaveSingleRawDataWhileScan_CheckBox,		SIGNAL(stateChanged(int)),		this,	SLOT(AutoSaveWhileScan_ChangeEvent(int)));
	connect(ui.AutoSaveMultiRawDataWhileScan_CheckBox,		SIGNAL(stateChanged(int)),		this,	SLOT(AutoSaveWhileScan_ChangeEvent(int)));
	connect(ui.AutoSaveImageWhileScan_CheckBox,				SIGNAL(stateChanged(int)),		this,	SLOT(AutoSaveWhileScan_ChangeEvent(int)));
	connect(ui.ScanButton,									SIGNAL(clicked()),				this,	SLOT(ScanOCTMode()));
	connect(ui.ScanOnceButton,								SIGNAL(clicked()),				this,	SLOT(ScanOCTOnceMode()));

	// OCT 測試
	connect(ui.RawDataToImage,								SIGNAL(clicked()),				this,	SLOT(ReadRawDataToImage()));
	connect(ui.EasyBorderDetect,							SIGNAL(clicked()),				this,	SLOT(ReadRawDataForBorderTest()));
	connect(ui.SingleImageShakeTestButton,					SIGNAL(clicked()),				this,	SLOT(ReadSingleRawDataForShakeTest()));
	connect(ui.MultiImageShakeTestButton,					SIGNAL(clicked()),				this,	SLOT(ReadMultiRawDataForShakeTest()));
	connect(ui.SlimLabViewRawData,							SIGNAL(clicked()),				this,	SLOT(SlimLabviewRawData()));

	// 點雲操作
	connect(ui.PCIndex,										SIGNAL(currentIndexChanged(int)),this,	SLOT(PCIndexChangeEvnet(int)));
	connect(ui.LoadPCButton,								SIGNAL(clicked()),				this,	SLOT(ReadPCEvent()));
	connect(ui.SavePCButton,								SIGNAL(clicked()),				this,	SLOT(SavePCEvent()));
	connect(ui.DeletePCButton,								SIGNAL(clicked()),				this,	SLOT(DeletePCEvent()));
	connect(ui.AlignLastTwoPCButton,						SIGNAL(clicked()),				this,	SLOT(AlignLastTwoPCEvent()));
	connect(ui.CombineLastTwoPCButton,						SIGNAL(clicked()),				this,	SLOT(CombineLastTwoPCEvent()));
	connect(ui.CombineAllPCButton,							SIGNAL(clicked()),				this,	SLOT(CombineAllPCEvent()));
	connect(ui.AlignmentAllPCTest,							SIGNAL(clicked()),				this,	SLOT(AlignmentAllPCTestEvent()));
	connect(ui.PassScanDataToPC,							SIGNAL(clicked()),				this,	SLOT(TransformMultiDataToPCEvent()));
	connect(ui.AveragePCErrorTest,							SIGNAL(clicked()),				this,	SLOT(AveragePCErrorTestEvent()));

	// Network 相關
	connect(ui.DataGenerationBtn,							SIGNAL(clicked()),				this,	SLOT(NetworkDataGenerateV2()));
	connect(ui.PredictResultTestingBtn,						SIGNAL(clicked()),				this,	SLOT(PredictResultTesting()));

	// Volumne Rendering Test
	//connect(ui.VolumeRenderingTestBtn,						SIGNAL(clicked()),				this,	SLOT(VolumeRenderTest()));

	// 顯示部分
	connect(ui.ScanNumSlider,								SIGNAL(valueChanged(int)),		this,	SLOT(ScanNumSlider_Change(int)));
	connect(UpdateGLTimer,									SIGNAL(timeout()),				this,	SLOT(DisplayPanelUpdate()));
	connect(ui.OCTViewDir,									SIGNAL(currentIndexChanged(int)),this,	SLOT(OCTViewOptionChange(int)));
	#pragma endregion
	#pragma region 初始化參數
	// UI 文字 & Scan Thread
	StartScanText = codec->toUnicode("掃    描    模    式\n(Start)");
	EndScanText = codec->toUnicode("掃    描    模    式\n(End)");

	// 存檔位置
	QString SaveLocation_Temp;
	QDate date = QDate::currentDate();
	
	QString currentDateStr = date.toString("yyyy.MM.dd");
	cout << "日期：" << currentDateStr.toStdString() << endl;
	#ifdef TEST_NO_OCT
	// 表示在桌機測試
	SaveLocation_Temp = "F:/OCT Scan DataSet/" + currentDateStr;
	#else
	// 表示在醫院測試
	SaveLocation_Temp = "V:/OCT Scan DataSet/" + currentDateStr;

	// 關閉一些進階功能
	ui.NetworkResult->setEnabled(false);
	ui.NetworkResultText->setEnabled(false);
	ui.OCTTestingBox->setEnabled(false);
	ui.BLETestingBox->setEnabled(false);
	ui.tabWidget->setTabEnabled(3, false);

	// BLE
	ui.BLEDeviceBox->setEnabled(false);
	#endif

	// 創建資料夾
	QDir().mkpath(SaveLocation_Temp);
	ui.SaveLocationText->setText(SaveLocation_Temp);

	// SegNet
	//segNetModel.Load(
	//	"./SegNetModel/segnet_inference.prototxt",				// prototxt
	//	"./SegNetModel/Models_iter_10000.caffemodel"			// caffemodel
	//);
	//segNetModel.ReshapeToMultiBatch(GPUBatchSize);

	// 更新 GL
	UpdateGLTimer->start(1.0f / 60);
	#pragma endregion
	#pragma region 傳 UI 指標進去
	// 藍芽的部分
	QVector<QObject*>		objList;

	objList.push_back(ui.BLEStatus);
	objList.push_back(ui.EularText);
	objList.push_back(this);
	objList.push_back(ui.BLEDeviceList);

	// OCT 顯示的部分
	objList.clear();
	objList.push_back(ui.ImageResult);
	objList.push_back(ui.BorderDetectionResult);
	objList.push_back(ui.NetworkResult);
	objList.push_back(ui.ScanNumSlider);
	objList.push_back(ui.ScanButton);
	objList.push_back(ui.SaveLocationText);
	objList.push_back(ui.DisplayPanel);
	objList.push_back(ui.PCIndex);
	objList.push_back(ui.OtherSideResult);
	objList.push_back(ui.NetworkResult_OtherSide);

	rawManager.SendUIPointer(objList);

	// 傳送 rawManager 到 OpenGL Widget
	ui.DisplayPanel->SetRawDataManager(&rawManager);
	#pragma endregion
	#pragma region TcpNetwork
	tcpSocket = new QTcpSocket(this);

	connect(tcpSocket,										SIGNAL(connected()),			this,	SLOT(TcpConnected()));
	connect(tcpSocket,										SIGNAL(disconnected()),			this,	SLOT(TcpDisConnected()));
	connect(tcpSocket,										SIGNAL(readyRead()),			this,	SLOT(TcpreadyRead()));
	#pragma endregion
}

// 九軸資料測試測試
void DentistProjectV2::PointCloudAlignmentTest()
{
	QVector<QString> FileInfo;
	QString GyroFileName = QFileDialog::getOpenFileName(this, codec->toUnicode("Gyro 檔案"), "E:/DentistData/ScanData/", "Gyro.txt", nullptr, QFileDialog::DontUseNativeDialog);
	if (GyroFileName != "")
	{
		QFile GyroFile(GyroFileName);
		GyroFile.open(QIODevice::ReadOnly);

		QTextStream ss(&GyroFile);
		QString TempFile;

		QDir currentDir(GyroFileName + "/../");

		float w, x, y, z;
		while (!ss.atEnd())
		{
			// 讀一條
			TempFile = ss.readLine();
			if(TempFile == "")
				break;

			// 拆開來 
			QStringList TempStr = TempFile.split(' ');
			RawDataType type = rawManager.ReadRawDataFromFileV2(currentDir.absoluteFilePath(TempStr[0]));
			rawManager.TransformToIMG(false);

			// Quat
			assert(TempStr.size() == 5 && "讀取的資料有誤!!");
			w = TempStr[1].toFloat();
			x = TempStr[2].toFloat();
			y = TempStr[3].toFloat();
			z = TempStr[4].toFloat();
			
			QQuaternion quat(w, x, y, z);
			rawManager.SavePointCloud(quat);
			rawManager.AlignmentPointCloud();
		}
		GyroFile.close();

		// 換圖片
		ui.ScanNumSlider->setEnabled(true);
		if (ui.ScanNumSlider->value() == 60)
			ScanNumSlider_Change(60);
		else
			ui.ScanNumSlider->setValue(60);

		// 更新面板
		ui.DisplayPanel->update();
	};
}

// OCT 相關(主要)
void DentistProjectV2::ChooseSaveLocaton()
{
	QString OCT_SaveLocation = QFileDialog::getExistingDirectory(this, "Save OCT Data Location", ui.SaveLocationText->text() + "/../", QFileDialog::DontUseNativeDialog);
	if (OCT_SaveLocation != "")
	{
		ui.SaveLocationText->setText(OCT_SaveLocation);

		// 創建目錄
		QDir().mkdir(OCT_SaveLocation);
	}
}
void DentistProjectV2::AutoSaveWhileScan_ChangeEvent(int signalNumber)
{
	if (ui.AutoSaveMultiRawDataWhileScan_CheckBox->isChecked())
	{
		QMessageBox::information(
			this,																												// 此視窗
			codec->toUnicode("貼心的提醒視窗"),																					// Title
			codec->toUnicode("如果勾選，會增加資料儲存致硬碟的時間")															// 中間的文字解說
		);
	}
}
void DentistProjectV2::ScanOCTMode()
{
	#ifdef TEST_NO_OCT
	// 判斷是否有
	QMessageBox::information(this, codec->toUnicode("目前無 OCT 裝置!!"), codec->toUnicode("請取消 Global Define!!"));

	// 這邊是確認檔名 OK 不 OK
	// 因為以前檔名有一個 Bug 導致會有 Error String 會有 Api Wait TimeOut (579) 的問題
	/*QFile TestFile(SaveLocation);
	if (!TestFile.open(QIODevice::WriteOnly))
		cout << "此檔名有問題!!" << endl;
	else
		cout << "此檔名沒有問題!!" << endl;
	TestFile.close();*/
	#else
	// 初始化變數
	bool NeedSave_Single_RawData = ui.AutoSaveSingleRawDataWhileScan_CheckBox->isChecked();
	bool NeedSave_Multi_RawData = ui.AutoSaveMultiRawDataWhileScan_CheckBox->isChecked();
	bool NeedSave_ImageData = ui.AutoSaveImageWhileScan_CheckBox->isChecked();
	bool AutoDeleteShakeData = ui.AutoDeleteShakeData_CheckBox->isChecked();

	// 判斷
	if (ui.ScanButton->text() == EndScanText)
	{
		if (!rawManager.bleManager.IsEstablished())
		{
			QMessageBox::information(this, codec->toUnicode("注意視窗"), codec->toUnicode("沒有連結九軸資訊!!"));
			return;
		}
		ui.ScanButton->setText(StartScanText);
		rawManager.SetScanOCTMode(true, &EndScanText, NeedSave_Single_RawData, NeedSave_Multi_RawData, NeedSave_ImageData, AutoDeleteShakeData);
	}
	else
		rawManager.SetScanOCTMode(false, &EndScanText, NeedSave_Single_RawData, NeedSave_Multi_RawData, NeedSave_ImageData, AutoDeleteShakeData);		// 設定只掃完最後一張就停止了
	#endif
}
void DentistProjectV2::ScanOCTOnceMode()
{
	#ifdef TEST_NO_OCT
	// 判斷是否有
	QMessageBox::information(this, codec->toUnicode("目前無 OCT 裝置!!"), codec->toUnicode("請取消 Global Define!!"));

	// 這邊是確認檔名 OK 不 OK
	// 因為以前檔名有一個 Bug 導致會有 Error String 會有 Api Wait TimeOut (579) 的問題
	/*QFile TestFile(SaveLocation);
	if (!TestFile.open(QIODevice::WriteOnly))
		cout << "此檔名有問題!!" << endl;
	else
		cout << "此檔名沒有問題!!" << endl;
	TestFile.close();*/
	#else
	rawManager.SetScanOCTOnceMode();
	#endif
}

// OCT 測試
void DentistProjectV2::ReadRawDataToImage()
{
	QString RawFileName = QFileDialog::getOpenFileName(this, codec->toUnicode("RawData 轉圖"), "E:/DentistData/ScanData/", "", nullptr, QFileDialog::DontUseNativeDialog);
	if (RawFileName != "")
	{
		RawDataType type = rawManager.ReadRawDataFromFileV2(RawFileName);
		rawManager.TransformToIMG(true);

		// UI 更改
		if (type == RawDataType::MULTI_DATA_TYPE)
		{
			rawManager.TransformToOtherSideView();

			QQuaternion quat;
			rawManager.SavePointCloud(quat);
			ui.ScanNumSlider->setEnabled(true);
		}
		else if (type == RawDataType::SINGLE_DATA_TYPE)
			ui.ScanNumSlider->setEnabled(false);

		// 換圖片
		if (ui.ScanNumSlider->value() == 60)
			ScanNumSlider_Change(60);
		else
			ui.ScanNumSlider->setValue(60);
		return;
	}

	// 其他狀況都需要進來這裡
	// Slider
	ui.ScanNumSlider->setEnabled(false);
	ui.ScanNumSlider->setValue(60);
}
void DentistProjectV2::ReadRawDataForBorderTest()
{
	QString RawFileName = QFileDialog::getOpenFileName(this, codec->toUnicode("邊界測試"), "E:/DentistData/ScanData/", "", nullptr, QFileDialog::DontUseNativeDialog);
	if (RawFileName != "")
	{
		RawDataType type = rawManager.ReadRawDataFromFileV2(RawFileName);
		rawManager.TransformToIMG(false);

		// UI 更改
		if (type == RawDataType::MULTI_DATA_TYPE)
		{
			rawManager.TransformToOtherSideView();

			QQuaternion quat;
			rawManager.SavePointCloud(quat);
			ui.ScanNumSlider->setEnabled(true);
		}
		else if (type == RawDataType::SINGLE_DATA_TYPE)
			ui.ScanNumSlider->setEnabled(false);

		// 換圖片
		if (ui.ScanNumSlider->value() == 60)
			ScanNumSlider_Change(60);
		else
			ui.ScanNumSlider->setValue(60);
		return;
	}

	// 其他狀況都需要進來這裡
	// Slider
	ui.ScanNumSlider->setEnabled(false);
	ui.ScanNumSlider->setValue(60);
}
void DentistProjectV2::ReadSingleRawDataForShakeTest()
{
	QStringList RawFileName = QFileDialog::getOpenFileNames(this, codec->toUnicode("晃動測式"), "E:/DentistData/ScanData/", "", nullptr, QFileDialog::DontUseNativeDialog);
	if (RawFileName.count() == 2)
	{
		RawDataType type = rawManager.ReadRawDataFromFileV2(RawFileName[0]);
		rawManager.TransformToIMG(false);

		// 直接給 250
		int* LastDataArray = NULL;
		rawManager.CopySingleBorder(LastDataArray);

		// 在讀下一筆資料
		rawManager.ReadRawDataFromFileV2(RawFileName[1]);
		rawManager.TransformToIMG(false);

		// 單張判斷
		rawManager.ShakeDetect_Single(LastDataArray, true);
		delete LastDataArray;
	}
	else
		cout << "請選擇兩張圖片!!" << endl;

	// 其他狀況都需要進來這裡
	// Slider
	ui.ScanNumSlider->setEnabled(false);
	ui.ScanNumSlider->setValue(60);
}
void DentistProjectV2::ReadMultiRawDataForShakeTest()
{
	//QMessageBox:: "未連結!!");
}
void DentistProjectV2::SlimLabviewRawData()
{
	QStringList RawFileNameList = QFileDialog::getOpenFileNames(this, codec->toUnicode("縮減 Labview 掃描的資料 (可 Shift 選取多筆資料)"), "E:/DentistData/ScanData/", "", nullptr, QFileDialog::DontUseNativeDialog);

	for (int i = 0; i < RawFileNameList.count(); i++)
	{
		// 開檔案
		QFile inputFile(RawFileNameList[i]);
		inputFile.open(QIODevice::ReadOnly);

		QFile slimFile(RawFileNameList[i] + "_slim");
		slimFile.open(QIODevice::WriteOnly);

		// 測試
		cout << RawFileNameList[i].toLocal8Bit().toStdString() << " "  << (inputFile.size()) << endl;
		QByteArray data = inputFile.readAll();


		// 錯誤判斷
		if (inputFile.size() != 819200000)
		{
			inputFile.close();
			slimFile.close();
			cout << "並非 Labview 掃描的檔案" << endl;
			continue;
		}

		QDataStream stream(&slimFile);
		stream.writeBytes(data, 512000000 - 4);

		//QDataStream
		inputFile.close();
		slimFile.close();
	}
}

// 點雲操作
void DentistProjectV2::PCIndexChangeEvnet(int)
{
	if (!rawManager.IsWidgetUpdate)
		rawManager.SelectIndex = ui.PCIndex->currentIndex();
}
void DentistProjectV2::ReadPCEvent()
{
	QStringList PointCloudFileList = QFileDialog::getOpenFileNames(this, codec->toUnicode("讀取點雲"), "E:/DentistData/PointCloud/", codec->toUnicode("點雲(*.xyz)"), nullptr, QFileDialog::DontUseNativeDialog);
	
	for (int i = 0; i < PointCloudFileList.count(); i++)
		if (PointCloudFileList[i] != "")
		{
			int index = rawManager.SelectIndex;
			PointCloudInfo info;
			info.ReadFromXYZ(PointCloudFileList[i]);
			rawManager.PointCloudArray.push_back(info);
			rawManager.InitRotationMarix.push_back(QMatrix4x4());

			// 更新相關設定
			rawManager.SelectIndex = rawManager.PointCloudArray.size() - 1;
			rawManager.PCWidgetUpdate();
		}
}
void DentistProjectV2::SavePCEvent()
{
	QString PointCloudFileName = QFileDialog::getSaveFileName(this, codec->toUnicode("邊界測試"), "E:/DentistData/PointCloud/", codec->toUnicode("點雲(*.xyz)"), nullptr, QFileDialog::DontUseNativeDialog);
	if (PointCloudFileName != "")
	{
		// 修正副檔名
		if (!PointCloudFileName.endsWith(".xyz"))
			PointCloudFileName += ".xyz";

		int index = rawManager.SelectIndex;
		rawManager.PointCloudArray[index].SaveXYZ(PointCloudFileName);
	}
}
void DentistProjectV2::DeletePCEvent()
{
	if (rawManager.SelectIndex < rawManager.PointCloudArray.size() && rawManager.SelectIndex >= 0)
	{
		// 刪除選擇的部分
		rawManager.PointCloudArray.erase(rawManager.PointCloudArray.begin() + rawManager.SelectIndex);
		if (rawManager.PointCloudArray.size() > rawManager.SelectIndex)
			rawManager.InitRotationMarix.erase(rawManager.InitRotationMarix.begin() + rawManager.SelectIndex);

		if (rawManager.SelectIndex == rawManager.PointCloudArray.size())			// 因為上面一行已經減去，所以這邊不用在減一
			rawManager.SelectIndex--;
		rawManager.PCWidgetUpdate();
	}
}
void DentistProjectV2::AlignLastTwoPCEvent()
{
	if (rawManager.PointCloudArray.size() >= 2)
		rawManager.AlignmentPointCloud();
}
void DentistProjectV2::CombineLastTwoPCEvent()
{
	if (rawManager.PointCloudArray.size() >= 2)
		rawManager.CombinePointCloud(rawManager.PointCloudArray.size() - 2, rawManager.PointCloudArray.size() - 1);
}
void DentistProjectV2::CombineAllPCEvent()
{
	if (rawManager.PointCloudArray.size() >= 2)
		for (int i = rawManager.PointCloudArray.size() - 1; i > 0; i--)
			rawManager.CombinePointCloud(0, i);
}
void DentistProjectV2::AlignmentAllPCTestEvent()
{
	QStringList PCList = QFileDialog::getOpenFileNames(this,
		codec->toUnicode("點雲的部分"),
		"E:/DentistData/PointCloud/",
		codec->toUnicode(""), nullptr,
		QFileDialog::DontUseNativeDialog);

	if (PCList.size() == 12)
		rawManager.TransformMultiDataToAlignment(PCList);
}
void DentistProjectV2::TransformMultiDataToPCEvent()
{
	QStringList RawDataList = QFileDialog::getOpenFileNames(this,
		codec->toUnicode("讀取 RawData"),
		"E:/DentistData/ScanData/2019.05.15/",
		codec->toUnicode(""), nullptr,
		QFileDialog::DontUseNativeDialog);

	if (RawDataList.size() == 13)
		rawManager.TransformMultiDataToPointCloud(RawDataList);
}
void DentistProjectV2::AveragePCErrorTestEvent() 
{
	rawManager.AverageErrorPC();
}

// Network 相關
void DentistProjectV2::NetworkDataGenerateV2()
{
	QString RawFileName = QFileDialog::getOpenFileName(this, codec->toUnicode("RawData 轉圖"), "E:/DentistData/ScanData/", "", nullptr, QFileDialog::DontUseNativeDialog);
	rawManager.NetworkDataGenerateV2(RawFileName);

	QQuaternion quat;
	rawManager.SavePointCloud(quat);

	// 換圖片
	ui.ScanNumSlider->setEnabled(true);
	if (ui.ScanNumSlider->value() == 60)
		ScanNumSlider_Change(60);
	else
		ui.ScanNumSlider->setValue(60);
}
void DentistProjectV2::PredictResultTesting()
{
	// 1. 先讀 Data
	ReadRawDataForBorderTest();

	// 2. 接著要抓出
	rawManager.NetworkDataGenerateInRamV2();
	if (!rawManager.CheckIsValidData())
	{
		cout << "Eigen 算有錯誤!!" << endl;
		return;
	}

	#ifdef USE_NETWORK_TO_PREDICT
	// 3. 存出所有Network需要圖片
	rawManager.SaveNetworkImage();

	// 4. 傳上伺服器Predict
	TcpNetwork();
	#else
	// 3. Python 預測資料
	//rawManager.PredictOtherSide();

	// 4. 預測整份的資料
	//rawManager.PredictFull();

	// 5. 把預測資料貼回原圖
	rawManager.LoadPredictImage();

	// 6. Smooth 結果並把點區塊刪除
	rawManager.SmoothNetworkData();

	// 7. 轉到 QImage 中
	rawManager.NetworkDataToQImage();

	// 8. 顯示結果
	rawManager.ShowImageIndex(60);
	#endif
}
void DentistProjectV2::TcpConnected()
{
	qDebug() << "socket connected";

	QString string = Requestmsg;
	tcpSocket->write(string.toLatin1());
	qDebug() << "Send: " << string;
}
void DentistProjectV2::TcpDisConnected()
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
		rawManager.NetworkDataToQImage();

		// 8. 顯示結果
		rawManager.ShowImageIndex(60);
	}
}
void DentistProjectV2::TcpreadyRead()
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
void DentistProjectV2::TcpNetwork()
{
	Requestmsg = "Sent";
	tcpSocket->connectToHost("140.118.175.94", 10000);
}
void DentistProjectV2::SentTest()
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
void DentistProjectV2::RecvTest(QByteArray buf)
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
void DentistProjectV2::useRAR()
{
	std::string AAA = "WinRAR.exe a -afzip Image.zip ./Predicts/*";

	system(AAA.c_str());
}
void DentistProjectV2::useUNRAR()
{
	std::string AAA = "WinRAR.exe -o+ x received_file.zip";

	system(AAA.c_str());
}

// Volume Render 測試
void DentistProjectV2::VolumeRenderTest()
{
	/*QString boungBoxPath = QFileDialog::getOpenFileName(this, codec->toUnicode("BoundingBox 檔案"), "E:/DentistData/NetworkData/2019.01.08 ToothBone1", "boundingBox.txt", nullptr, QFileDialog::DontUseNativeDialog);
	rawManager.ImportVolumeDataTest(boungBoxPath);*/
}

// 顯示部分的事件
void DentistProjectV2::ScanNumSlider_Change(int value)
{
	rawManager.ShowImageIndex(value);
	ui.ScanNum_Value->setText(QString::number(value));
}
void DentistProjectV2::DisplayPanelUpdate()
{
	ui.DisplayPanel->update();
}
void DentistProjectV2::OCTViewOptionChange(int)
{
	OpenGLWidget* widget = ui.DisplayPanel;
	widget->OCTViewType = ui.OCTViewDir->currentIndex();
}