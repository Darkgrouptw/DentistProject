#include "DentistProject.h"

DentistProject::DentistProject(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);

	#pragma region 事件連結
	// 事件連結
	connect(ui.actionLoadSTL,								SIGNAL(triggered()),			this,	SLOT(LoadSTL()));

	// 顯示事件
	connect(ui.RenderTriangle_CheckBox,						SIGNAL(clicked()),				this,	SLOT(SetRenderTriangleBool()));
	connect(ui.RenderBorder_CheckBox,						SIGNAL(clicked()),				this,	SLOT(SetRenderBorderBool()));
	connect(ui.RenderPointDot_CheckBox,						SIGNAL(clicked()),				this,	SLOT(SetRenderPointCloudBool()));

	// 藍芽部分
	connect(ui.BtnSearchCom,								SIGNAL(clicked()),				this,	SLOT(SearchCOM()));
	connect(ui.BtnConnectCOM,								SIGNAL(clicked()),				this,	SLOT(ConnectCOM()));
	connect(ui.BtnScanBLEDevice,							SIGNAL(clicked()),				this,	SLOT(ScanBLEDevice()));
	connect(ui.BtnConnectBLEDevice,							SIGNAL(clicked()),				this,	SLOT(ConnectBLEDeivce()));
	
	// OCT 相關(主要)
	connect(ui.SaveLocationBtn,								SIGNAL(clicked()),				this,	SLOT(ChooseSaveLocaton()));
	connect(ui.SaveWithTime_CheckBox,						SIGNAL(stateChanged(int)),		this,	SLOT(SaveWithTime_ChangeEvent(int)));
	connect(ui.AutoScanRawDataWhileScan_CheckBox,			SIGNAL(stateChanged(int)),		this,	SLOT(AutoSaveWhileScan_ChangeEvent(int)));
	connect(ui.AutoScanImageWhileScan_CheckBox,				SIGNAL(stateChanged(int)),		this,	SLOT(AutoSaveWhileScan_ChangeEvent(int)));
	connect(ui.ScanButton,									SIGNAL(clicked()),				this,	SLOT(ScanOCT()));

	// OCT 測試
	connect(ui.RawDataToImage,								SIGNAL(clicked()),				this,	SLOT(ReadRawDataToImage()));
	connect(ui.RawDataCheck,								SIGNAL(clicked()),				this,	SLOT(WriteRawDataForTesting()));
	connect(ui.EasyBorderDetect,							SIGNAL(clicked()),				this,	SLOT(ReadRawDataForBorderTest()));
	connect(ui.BeepSoundTestButton,							SIGNAL(clicked()),				this,	SLOT(BeepSoundTest()));
	connect(ui.ShakeTestButton,								SIGNAL(clicked()),				this,	SLOT(ReadRawDataForShakeTest()));
	connect(ui.SegNetTestButton,							SIGNAL(clicked()),				this,	SLOT(SegNetTest()));

	// 顯示部分
	connect(ui.ScanNumSlider,								SIGNAL(valueChanged(int)),		this,	SLOT(ScanNumSlider_Change(int)));
	#pragma endregion
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
	objList.push_back(ui.NetworkResult);
	objList.push_back(ui.FinalResult);

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
	#endif

	// 創建資料夾
	QDir().mkpath(SaveLocation_Temp);
	ui.SaveLocationText->setText(SaveLocation_Temp);

	// SegNet
	segNetModel.Load(
		"./SegNetModel/segnet_inference.prototxt",				// prototxt
		"./SegNetModel/Models_iter_10000.caffemodel"			// caffemodel
	);
	segNetModel.ReshapeToMultiBatch(GPUBatchSize);
	#pragma endregion
}

// 其他事件
void DentistProject::LoadSTL()
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
void DentistProject::ScanData()
{
	cout << "Test Scan" << endl;
}

// Render Options
void DentistProject::SetRenderTriangleBool()
{
	ui.DisplayPanel->SetRenderTriangleBool(ui.RenderTriangle_CheckBox->isChecked());
	ui.DisplayPanel->update();
}
void DentistProject::SetRenderBorderBool()
{
	ui.DisplayPanel->SetRenderBorderBool(ui.RenderBorder_CheckBox->isChecked());
	ui.DisplayPanel->update();
}
void DentistProject::SetRenderPointCloudBool()
{
	ui.DisplayPanel->SetRenderPointCloudBool(ui.RenderPointDot_CheckBox->isChecked());
	ui.DisplayPanel->update();
}

// 藍芽事件
void DentistProject::SearchCOM()
{
	QStringList COMListArray = rawManager.bleManager.GetCOMPortsArray();
	ui.COMList->clear();
	ui.COMList->addItems(COMListArray);
}
void DentistProject::ConnectCOM()
{
	rawManager.bleManager.Initalize(ui.COMList->currentText());
}
void DentistProject::ScanBLEDevice()
{
	if (rawManager.bleManager.IsInitialize())
		rawManager.bleManager.Scan();
	else
	{
		// 這邊要跳出說沒有建立連線，所以不能搜尋
	}
}
void DentistProject::ConnectBLEDeivce()
{
	rawManager.bleManager.Connect(ui.BLEDeviceList->currentIndex());
}

// OCT 相關(主要)
void DentistProject::ChooseSaveLocaton()
{
	QString OCT_SaveLocation = QFileDialog::getExistingDirectory(this, "Save OCT Data Location", ui.SaveLocationText->text() + "/../", QFileDialog::DontUseNativeDialog);
	if (OCT_SaveLocation != "")
	{
		ui.SaveLocationText->setText(OCT_SaveLocation);

		// 創建目錄
		QDir().mkdir(OCT_SaveLocation);
	}
}
void DentistProject::SaveWithTime_ChangeEvent(int signalNumber)
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
void DentistProject::AutoSaveWhileScan_ChangeEvent(int signalNumber)
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
void DentistProject::ScanOCT()
{
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
	/*QFile TestFile(SaveLocation);
	if (!TestFile.open(QIODevice::WriteOnly))
		cout << "此檔名有問題!!" << endl;
	else
		cout << "此檔名沒有問題!!" << endl;
	TestFile.close();*/
	return;
	#else
	// 開始 Scan

	bool NeedSave_RawData = ui.AutoScanRawDataWhileScan_CheckBox->isChecked();
	bool NeedSave_Image = ui.AutoScanImageWhileScan_CheckBox->isChecked();
	while (true)
	{
		rawManager.ScanDataFromDevice(SaveLocation, NeedSave_RawData);
		rawManager.TranformToIMG(NeedSave_Image);
		if (rawManager.ShakeDetect(this, false))
		{
			// 沒晃動到
			cout << "沒有晃動到" << endl;
			break;
		}
		else
			cout << "晃到重拍" << endl;

	}
	#endif
	#pragma endregion
	#pragma region 網路預測結果
	//rawManager.GenerateNetworkData();
	#pragma endregion
}

// OCT 測試
void DentistProject::ReadRawDataToImage()
{
	QString RawFileName = QFileDialog::getOpenFileName(this, "Read Raw Data", "D:/Dentist/Data/ScanData/", "", nullptr, QFileDialog::DontUseNativeDialog);
	if (RawFileName != "")
	{
		rawManager.ReadRawDataFromFile(RawFileName);
		rawManager.TranformToIMG(true);

		// UI 更改
		ui.ScanNumSlider->setEnabled(true);
		if (ui.ScanNumSlider->value() == 60)
			ScanNumSlider_Change(60);
		else
			ui.ScanNumSlider->setValue(60);
	}
	else
	{
		// Slider
		ui.ScanNumSlider->setEnabled(false);
		ui.ScanNumSlider->setValue(60);
	}
}
void DentistProject::ReadRawDataForBorderTest()
{
	QString RawFileName = QFileDialog::getOpenFileName(this, "Read Raw Data", "D:/Dentist/Data/ScanData/", "", nullptr, QFileDialog::DontUseNativeDialog);
	if (RawFileName != "")
	{
		rawManager.ReadRawDataFromFile(RawFileName);
		rawManager.TranformToIMG(false);

		// UI 更改
		ui.ScanNumSlider->setEnabled(true);
		if (ui.ScanNumSlider->value() == 60)
			ScanNumSlider_Change(60);
		else
			ui.ScanNumSlider->setValue(60);
	}
	else
	{
		// Slider
		ui.ScanNumSlider->setEnabled(false);
		ui.ScanNumSlider->setValue(60);
	}
}
void DentistProject::WriteRawDataForTesting()
{
	QString RawFileName = QFileDialog::getOpenFileName(this, "Read Raw Data", "D:/Dentist/Data/ScanData/", "", nullptr, QFileDialog::DontUseNativeDialog);
	if (RawFileName != "")
	{
		rawManager.ReadRawDataFromFile(RawFileName);
		rawManager.WriteRawDataToFile("./Images/OCTImages/rawdata_v2/");
	}
}
void DentistProject::ReadRawDataForShakeTest()
{
	QString RawFileName = QFileDialog::getOpenFileName(this, "Read Raw Data", "D:/Dentist/Data/ScanData/", "", nullptr, QFileDialog::DontUseNativeDialog);
	if (RawFileName != "")
	{
		rawManager.ReadRawDataFromFile(RawFileName);
		rawManager.TranformToIMG(false);
		cout << (rawManager.ShakeDetect(this, true) ? "無晃動!!" : "有晃動!!") << endl;

		// UI 更改
		ui.ScanNumSlider->setEnabled(true);
		if (ui.ScanNumSlider->value() == 60)
			ScanNumSlider_Change(60);
		else
			ui.ScanNumSlider->setValue(60);
	}
	else
	{
		// Slider
		ui.ScanNumSlider->setEnabled(false);
		ui.ScanNumSlider->setValue(60);
	}
}
void DentistProject::BeepSoundTest()
{
	cout << "\a";
}
void DentistProject::SegNetTest()
{
	QString RawFileName = QFileDialog::getOpenFileName(this, "Read Raw Data", "D:/Dentist/Data/ScanData/", "", nullptr, QFileDialog::DontUseNativeDialog);
	if (RawFileName != "")
	{
		rawManager.ReadRawDataFromFile(RawFileName);
		rawManager.TranformToIMG(false);
		QVector<Mat> Data = rawManager.GenerateNetworkData();

		// 預測
		vector<Mat> PredictArray = segNetModel.Predict(Data.toStdVector());
		for (int i = 0; i < PredictArray.size(); i++)
			PredictArray[i] = segNetModel.Visualization(PredictArray[i]);

		// 傳回去顯示
		QVector<Mat> qPredictArray = QVector<Mat>::fromStdVector(PredictArray);
		for (int i = 0; i < qPredictArray.size(); i+=5)
		{
			for (int j = 0; j < 5; j++)
			{
				QString Loc = "D:/aaaaaa/" + QString::number(i / 5 + 60) + "_" + QString::number(j) + ".png";
				imwrite(Loc.toStdString(), qPredictArray[i]);
			}
		}
		//rawManager.SetPredictData(qPredictArray);

		// UI 更改
		ui.ScanNumSlider->setEnabled(true);
		if (ui.ScanNumSlider->value() == 60)
			ScanNumSlider_Change(60);
		else
			ui.ScanNumSlider->setValue(60);
	}
	else
	{
		// Slider
		ui.ScanNumSlider->setEnabled(false);
		ui.ScanNumSlider->setValue(60);
	}
}

// 顯示部分的事件
void DentistProject::ScanNumSlider_Change(int value)
{
	rawManager.ShowImageIndex(value);
	ui.ScanNum_Value->setText(QString::number(value));
}
