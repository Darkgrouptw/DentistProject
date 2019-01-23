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
	//connect(ui.ScanButton,									SIGNAL(clicked()),				this,	SLOT(ScanOCT()));

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

	rawManager.SendUIPointer(objList);

	// 傳送 rawManager 到 OpenGL Widget
	ui.DisplayPanel->SetRawDataManager(&rawManager);
	#pragma endregion
	#pragma region 初始化參數
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
	//cout << ui.SaveWithTime_CheckBox->isChecked() << endl;
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
