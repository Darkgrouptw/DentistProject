#include "DentistProjectV2.h"

DentistProjectV2::DentistProjectV2(QWidget *parent) : QMainWindow(parent)
{
	ui.setupUi(this);
	#pragma region 事件連結
	// 事件連結
	//connect(ui.actionLoadSTL,								SIGNAL(triggered()),			this,	SLOT(LoadSTL()));

	//// 顯示事件
	//connect(ui.RenderTriangle_CheckBox,						SIGNAL(clicked()),				this,	SLOT(SetRenderTriangleBool()));
	//connect(ui.RenderBorder_CheckBox,						SIGNAL(clicked()),				this,	SLOT(SetRenderBorderBool()));
	//connect(ui.RenderPointDot_CheckBox,						SIGNAL(clicked()),				this,	SLOT(SetRenderPointCloudBool()));

	//// 藍芽部分
	//connect(ui.BtnSearchCom,								SIGNAL(clicked()),				this,	SLOT(SearchCOM()));
	//connect(ui.BtnConnectCOM,								SIGNAL(clicked()),				this,	SLOT(ConnectCOM()));
	//connect(ui.BtnScanBLEDevice,							SIGNAL(clicked()),				this,	SLOT(ScanBLEDevice()));
	//connect(ui.BtnConnectBLEDevice,							SIGNAL(clicked()),				this,	SLOT(ConnectBLEDeivce()));
	//connect(ui.ResetRotationMode,							SIGNAL(clicked()),				this,	SLOT(SetRotationMode()));
	//connect(ui.GyroscopeResetToZero,						SIGNAL(clicked()),				this,	SLOT(GyroResetToZero()));
	//
	//// 藍芽測試
	//connect(ui.PointCloudAlignmentTestBtn,					SIGNAL(clicked()),				this,	SLOT(PointCloudAlignmentTest()));

	// OCT 相關(主要)
	connect(ui.SaveLocationBtn,								SIGNAL(clicked()),				this,	SLOT(ChooseSaveLocaton()));
	connect(ui.SaveWithTime_CheckBox,						SIGNAL(stateChanged(int)),		this,	SLOT(SaveWithTime_ChangeEvent(int)));
	connect(ui.AutoScanRawDataWhileScan_CheckBox,			SIGNAL(stateChanged(int)),		this,	SLOT(AutoSaveWhileScan_ChangeEvent(int)));
	connect(ui.AutoScanImageWhileScan_CheckBox,				SIGNAL(stateChanged(int)),		this,	SLOT(AutoSaveWhileScan_ChangeEvent(int)));
	connect(ui.ScanButton,									SIGNAL(clicked()),				this,	SLOT(ScanOCTMode()));

	// OCT 測試
	connect(ui.RawDataToImage,								SIGNAL(clicked()),				this,	SLOT(ReadRawDataToImage()));
	connect(ui.EasyBorderDetect,							SIGNAL(clicked()),				this,	SLOT(ReadRawDataForBorderTest()));
	//connect(ui.BeepSoundTestButton,							SIGNAL(clicked()),				this,	SLOT(BeepSoundTest()));
	//connect(ui.ShakeTestButton,								SIGNAL(clicked()),				this,	SLOT(ReadRawDataForShakeTest()));
	//connect(ui.SegNetTestButton,							SIGNAL(clicked()),				this,	SLOT(SegNetTest()));

	//// 點雲 Render 相關
	//connect(ui.PrePCBtn,									SIGNAL(clicked()),				this,	SLOT(PrePointCloudClick()));
	//connect(ui.NextPCBtn,									SIGNAL(clicked()),				this,	SLOT(NextPointCloudClick()));

	// 顯示部分
	connect(ui.ScanNumSlider,								SIGNAL(valueChanged(int)),		this,	SLOT(ScanNumSlider_Change(int)));
	#pragma endregion
	#pragma region 初始化參數
	// UI 文字 & Scan Thread
	StartScanText = codec->toUnicode("掃    描    模    式\n(Start)");
	EndStartText = codec->toUnicode("掃    描    模    式\n(End)");

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
	#pragma endregion
	#pragma region 傳 UI 指標進去
	// 藍芽的部分
	QVector<QObject*>		objList;

	objList.push_back(ui.BLEStatus);
	objList.push_back(ui.EularText);
	objList.push_back(this);
	objList.push_back(ui.BLEDeviceList);
	
	rawManager.bleManager.SendUIPointer(objList);

	// OCT 顯示的部分
	objList.clear();
	objList.push_back(ui.ImageResult);
	objList.push_back(ui.BorderDetectionResult);
	objList.push_back(ui.NetworkResult);
	objList.push_back(ui.ScanButton);
	objList.push_back((QObject*)&EndStartText);

	rawManager.SendUIPointer(objList);

	// 傳送 rawManager 到 OpenGL Widget
	ui.DisplayPanel->SetRawDataManager(&rawManager);
	#pragma endregion
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
void DentistProjectV2::SaveWithTime_ChangeEvent(int signalNumber)
{
	if (!ui.SaveWithTime_CheckBox->isChecked())
	{
		QMessageBox::information(
			this,																												// 此視窗
			codec->toUnicode("貼心的提醒視窗"),																					// Title
			codec->toUnicode("如果取消勾選，那儲存位置會以掃描順序來定\n(Ex: V:/OCT OCT Scan DataSet/1)")						// 中間的文字解說
		);
	}
}
void DentistProjectV2::AutoSaveWhileScan_ChangeEvent(int signalNumber)
{
	if (ui.AutoScanRawDataWhileScan_CheckBox->isChecked())
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
	//cout << ui.ScanButton->text().toStdString() << endl;
	//cout << EndStartText.toStdString() << endl;
	if (ui.ScanButton->text() == EndStartText)
	{
		ui.ScanButton->setText(StartScanText);
		//rawManager.Init
		rawManager.SetScanOCTMode(true);
	}
	else
	{
		rawManager.SetScanOCTMode(false);
		//rawManager.SetScanOCTMode(false);
	}
	/*
	#pragma region 檔名處理
	QString SaveLocation;							// 最後儲存的路徑
	if (ui.SaveWithTime_CheckBox->isChecked())
	{
		QTime currentTime = QTime::currentTime();
		QString TimeFileName = currentTime.toString("hh_mm_ss_zzz");
		cout << "現在時間: " << TimeFileName.toStdString() << endl;

		SaveLocation = QDir(ui.SaveLocationText->text()).absoluteFilePath(TimeFileName);
	}
	else
	{
		SaveLocation = QDir(ui.SaveLocationText->text()).absoluteFilePath(QString::number(ScanIndex));
		ScanIndex++;
	}
	cout << "儲存位置: " << SaveLocation.toStdString() << endl;
	#pragma endregion
	#pragma region 掃描
	#ifdef TEST_NO_OCT
	// 判斷是否有
	QMessageBox::information(this, codec->toUnicode("目前無 OCT 裝置!!"), codec->toUnicode("請取消 Global Define!!"));

	// 這邊是確認檔名 OK 不 OK
	// 因為以前檔名有一個 Bug 導致會有 Error String 會有 Api Wait TimeOut (579) 的問題
	//QFile TestFile(SaveLocation);
	//if (!TestFile.open(QIODevice::WriteOnly))
	//	cout << "此檔名有問題!!" << endl;
	//else
	//	cout << "此檔名沒有問題!!" << endl;
	//TestFile.close();
	return;
	#else
	// 開始 Scan

	bool NeedSave_RawData = ui.AutoScanRawDataWhileScan_CheckBox->isChecked();
	bool NeedSave_Image = ui.AutoScanImageWhileScan_CheckBox->isChecked();
	//while (true)
	//{
	//	rawManager.ScanDataFromDevice(SaveLocation, NeedSave_RawData);
	//	rawManager.TranformToIMG(NeedSave_Image);
	//	if (rawManager.ShakeDetect(this, false))
	//	{
	//		// 沒晃動到
	//		cout << "沒有晃動到" << endl;
	//		break;
	//	}
	//	else
	//		cout << "晃到重拍" << endl;

	//}
	//rawManager.ScanDataFromDevice(SaveLocation, NeedSave_RawData);
	rawManager.ScanSingleData(SaveLocation, NeedSave_RawData);
	//rawManager.TranformToIMG(NeedSave_Image);
	#endif
	#pragma endregion
	#pragma region 網路預測結果
	//rawManager.GenerateNetworkData();
	#pragma endregion
	*/
}

// OCT 測試
void DentistProjectV2::ReadRawDataToImage()
{
	QString RawFileName = QFileDialog::getOpenFileName(this, "Read Raw Data", "D:/Dentist/Data/ScanData/", "", nullptr, QFileDialog::DontUseNativeDialog);
	if (RawFileName != "")
	{
		RawDataType type = rawManager.ReadRawDataFromFileV2(RawFileName);
		rawManager.TranformToIMG(true);

		// UI 更改
		if (type == RawDataType::MULTI_DATA_TYPE)
		{
			ui.ScanNumSlider->setEnabled(true);
			if (ui.ScanNumSlider->value() == 60)
				ScanNumSlider_Change(60);
			else
				ui.ScanNumSlider->setValue(60);
			return;
		}
	}

	// 其他狀況都需要進來這裡
	// Slider
	ui.ScanNumSlider->setEnabled(false);
	ui.ScanNumSlider->setValue(60);
}
void DentistProjectV2::ReadRawDataForBorderTest()
{
	QString RawFileName = QFileDialog::getOpenFileName(this, "Read Raw Data", "D:/Dentist/Data/ScanData/", "", nullptr, QFileDialog::DontUseNativeDialog);
	if (RawFileName != "")
	{
		RawDataType type = rawManager.ReadRawDataFromFileV2(RawFileName);
		rawManager.TranformToIMG(false);

		// UI 更改
		if (type == RawDataType::MULTI_DATA_TYPE)
		{
			ui.ScanNumSlider->setEnabled(true);
			if (ui.ScanNumSlider->value() == 60)
				ScanNumSlider_Change(60);
			else
				ui.ScanNumSlider->setValue(60);
			return;
		}
		else if (type == RawDataType::SINGLE_DATA_TYPE)
		{
			ui.ScanNumSlider->setEnabled(false);
			ScanNumSlider_Change(0);
			return;
		}
	}

	// 其他狀況都需要進來這裡
	// Slider
	ui.ScanNumSlider->setEnabled(false);
	ui.ScanNumSlider->setValue(60);
}

// 顯示部分的事件
void DentistProjectV2::ScanNumSlider_Change(int value)
{
	rawManager.ShowImageIndex(value);
	ui.ScanNum_Value->setText(QString::number(value));
}
