#include "DentistTrainningDataMaker.h"

DentistTrainningDataMaker::DentistTrainningDataMaker(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);

	// 事件連結
	connect(ui.actionLoadSTL,				SIGNAL(triggered()),			this,	SLOT(LoadSTL()));
	//connect(ui.TestFullScan,				SIGNAL(triggered()),			this,	SLOT(ScanData()));

	// 顯示事件
	connect(ui.RenderTriangle_CheckBox,		SIGNAL(clicked()),				this,	SLOT(SetRenderTriangleBool()));
	connect(ui.RenderBorder_CheckBox,		SIGNAL(clicked()),				this,	SLOT(SetRenderBorderBool()));
	connect(ui.RenderPointDot_CheckBox,		SIGNAL(clicked()),				this,	SLOT(SetRenderPointCloudBool()));

	// 藍芽部分
	connect(ui.BtnSearchCom,				SIGNAL(clicked()),				this,	SLOT(SearchCOM()));
	connect(ui.BtnConnectCOM,				SIGNAL(clicked()),				this,	SLOT(ConnectCOM()));
	connect(ui.BtnScanBLEDevice,			SIGNAL(clicked()),				this,	SLOT(ScanBLEDevice()));
	connect(ui.BtnConnectBLEDevice,			SIGNAL(clicked()),				this,	SLOT(ConnectBLEDeivce()));
	
	// OCT 相關
	connect(ui.RawDataToImage,				SIGNAL(clicked()),				this,	SLOT(ReadRawDataToImage()));
	connect(ui.EasyBorderDetect,			SIGNAL(clicked()),				this,	SLOT(ReadRawDataForBorderTest()));
	connect(ui.SaveLocationBtn,				SIGNAL(clicked()),				this,	SLOT(ChooseSaveLocaton()));
	connect(ui.SaveWithTime_CheckBox,		SIGNAL(stateChanged(int)),		this,	SLOT(SaveWithTime_ChangeEvent(int)));
	connect(ui.AutoScanWhileScan_CheckBox,	SIGNAL(stateChanged(int)),		this,	SLOT(AutoSaveWhileScan_ChangeEvent(int)));
	connect(ui.ScanButton,					SIGNAL(clicked()),				this,	SLOT(ScanOCT()));
	connect(ui.BeepSoundTest,				SIGNAL(clicked()),				this,	SLOT(BeepSoundTest()));

	// 顯示部分
	connect(ui.ScanNumSlider,				SIGNAL(valueChanged(int)),		this,	SLOT(ScanNumSlider_Change(int)));

	// 其他
	#pragma region 傳 UI 指標進去
	// 藍芽的部分
	QVector<QObject*>		objList;

	objList.push_back(ui.BLEStatus);
	objList.push_back(ui.QuaternionText);
	objList.push_back(this);
	objList.push_back(ui.BLEDeviceList);
	
	rawManager.bleManager.SendUIPointer(objList);

	// OCT 顯示的部分
	objList.clear();
	objList.push_back(ui.ImageResult);
	objList.push_back(ui.FinalResult);

	rawManager.SendUIPointer(objList);
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
	#endif

	// 創建資料夾
	QDir().mkpath(SaveLocation_Temp);
	ui.SaveLocationText->setText(SaveLocation_Temp);
	#pragma endregion
}

// 其他事件
void DentistTrainningDataMaker::LoadSTL()
{
	QString STLFileName = QFileDialog::getOpenFileName(this, "Load STL", "./STLs", "STL (*.stl)", nullptr, QFileDialog::DontUseNativeDialog);
	system("cls");
	cout << "讀取位置：" << STLFileName.toStdString() << endl;

	if (STLFileName != "")
	{
		ui.DisplayPanel->LoadSTLFile(STLFileName);
		ui.DisplayPanel->update();
	}
}
void DentistTrainningDataMaker::ScanData()
{
	cout << "Test Scan" << endl;
}

// Render Options
void DentistTrainningDataMaker::SetRenderTriangleBool()
{
	ui.DisplayPanel->SetRenderTriangleBool(ui.RenderTriangle_CheckBox->isChecked());
	ui.DisplayPanel->update();
}
void DentistTrainningDataMaker::SetRenderBorderBool()
{
	ui.DisplayPanel->SetRenderBorderBool(ui.RenderBorder_CheckBox->isChecked());
	ui.DisplayPanel->update();
}
void DentistTrainningDataMaker::SetRenderPointCloudBool()
{
	ui.DisplayPanel->SetRenderPointCloudBool(ui.RenderPointDot_CheckBox->isChecked());
	ui.DisplayPanel->update();
}

// 藍芽事件
void DentistTrainningDataMaker::SearchCOM()
{
	QStringList COMListArray = rawManager.bleManager.GetCOMPortsArray();
	ui.COMList->clear();
	ui.COMList->addItems(COMListArray);
}
void DentistTrainningDataMaker::ConnectCOM()
{
	rawManager.bleManager.Initalize(ui.COMList->currentText());
}
void DentistTrainningDataMaker::ScanBLEDevice()
{
	if (rawManager.bleManager.IsInitialize())
		rawManager.bleManager.Scan();
	else
	{
		// 這邊要跳出說沒有建立連線，所以不能搜尋
	}
}
void DentistTrainningDataMaker::ConnectBLEDeivce()
{
	rawManager.bleManager.Connect(ui.BLEDeviceList->currentIndex());
}

// OCT 相關
void DentistTrainningDataMaker::ReadRawDataToImage()
{
	QString RawFileName = QFileDialog::getOpenFileName(this, "Read Raw Data", "D:/Dentist/Data/ScanData/", "", nullptr, QFileDialog::DontUseNativeDialog);
	if (RawFileName != "")
	{
		rawManager.ReadRawDataFromFile(RawFileName);
		rawManager.TranformToIMG(false);

		// UI 更改
		ui.ScanNumSlider->setEnabled(true);
		if (ui.ScanNumSlider->value() == 0)
			ScanNumSlider_Change(0);
		else
			ui.ScanNumSlider->setValue(0);
	}
	else
	{
		// Slider
		ui.ScanNumSlider->setEnabled(false);
		ui.ScanNumSlider->setValue(60);
	}
}
void DentistTrainningDataMaker::ReadRawDataForBorderTest()
{
	QString RawFileName = QFileDialog::getOpenFileName(this, "Read Raw Data", "D:/Dentist/Data/ScanData/", "", nullptr, QFileDialog::DontUseNativeDialog);
	if (RawFileName != "")
	{
		rawManager.ReadRawDataFromFile(RawFileName);
		rawManager.TranformToIMG(false);
		cout << (rawManager.ShakeDetect(this, true) ? "無晃動!!" : "有晃動!!") << endl;

		// UI 更改
		ui.ScanNumSlider->setEnabled(true);
		if (ui.ScanNumSlider->value() == 0)
			ScanNumSlider_Change(0);
		else
			ui.ScanNumSlider->setValue(0);
	}
	else
	{
		// Slider
		ui.ScanNumSlider->setEnabled(false);
		ui.ScanNumSlider->setValue(60);
	}
}
void DentistTrainningDataMaker::ChooseSaveLocaton()
{
	QString OCT_SaveLocation = QFileDialog::getExistingDirectory(this, "Save OCT Data Location", ui.SaveLocationText->text() + "/../", QFileDialog::DontUseNativeDialog);
	if (OCT_SaveLocation != "")
	{
		ui.SaveLocationText->setText(OCT_SaveLocation);

		// 創建目錄
		QDir().mkdir(OCT_SaveLocation);
	}
}
void DentistTrainningDataMaker::SaveWithTime_ChangeEvent(int signalNumber)
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
void DentistTrainningDataMaker::AutoSaveWhileScan_ChangeEvent(int signalNumber)
{
	if (ui.AutoScanWhileScan_CheckBox->isChecked())
	{
		QMessageBox::information(
			this,																												// 此視窗
			codec->toUnicode("貼心的提醒視窗"),																					// Title
			codec->toUnicode("如果勾選，會增加資料儲存致硬碟的時間")															// 中間的文字解說
		);
	}
	//cout << ui.SaveWithTime_CheckBox->isChecked() << endl;
}
void DentistTrainningDataMaker::ScanOCT()
{
	#pragma region 檔名處理
	QString SaveLocation;							// 最後儲存的路徑
	if (ui.SaveWithTime_CheckBox->isChecked())
	{
		QTime currentTime = QTime::currentTime();
		QString TimeFileName = currentTime.toString("hh:mm:ss.zzz");
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
	//QString location = 

	// 掃描
	#ifdef TEST_NO_OCT
	// 判斷是否有
	QMessageBox::information(this, codec->toUnicode("目前無 OCT 裝置!!"), codec->toUnicode("請取消 Global Define!!"));
	#else
	// 開始 Scan

	//ui.SaveLocationText

	bool NeedSave = ui.AutoScanWhileScan_CheckBox->isChecked();
	while (true)
	{
		rawManager.ScanDataFromDevice(SaveLocation, NeedSave);
		rawManager.TranformToIMG(NeedSave);
		if (rawManager.ShakeDetect(this, true))
		{
			// 沒晃動到
			cout << "沒有晃動到" << endl;
			break;
		}
		else
			cout << "晃到重拍" << endl;

	}
	#endif
}
void DentistTrainningDataMaker::BeepSoundTest()
{
	cout << "\a";
}

// 顯示部分的事件
void DentistTrainningDataMaker::ScanNumSlider_Change(int value)
{
	rawManager.ShowImageIndex(value);
	ui.ScanNum_Value->setText(QString::number(value));
}
