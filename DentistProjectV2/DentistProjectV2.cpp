#include "DentistProjectV2.h"

DentistProjectV2::DentistProjectV2(QWidget *parent) : QMainWindow(parent)
{
	ui.setupUi(this);
	#pragma region UpdateGLTimer
	UpdateGLTimer = new QTimer();
	#pragma endregion
	#pragma region 事件連結
	// 藍芽部分
	connect(ui.BtnSearchCom,								SIGNAL(clicked()),				this,	SLOT(SearchCOM()));
	connect(ui.BtnConnectCOM,								SIGNAL(clicked()),				this,	SLOT(ConnectCOM()));
	connect(ui.BtnScanBLEDevice,							SIGNAL(clicked()),				this,	SLOT(ScanBLEDevice()));
	connect(ui.BtnConnectBLEDevice,							SIGNAL(clicked()),				this,	SLOT(ConnectBLEDeivce()));
	connect(ui.ResetRotationMode,							SIGNAL(clicked()),				this,	SLOT(SetRotationMode()));
	connect(ui.GyroscopeResetToZero,						SIGNAL(clicked()),				this,	SLOT(GyroResetToZero()));
	connect(ui.BLEConnect_OneBtn,							SIGNAL(clicked()),				this,	SLOT(ConnectBLEDevice_OneBtn()));
	
	// 藍芽測試
	connect(ui.PointCloudAlignmentTestBtn,					SIGNAL(clicked()),				this,	SLOT(PointCloudAlignmentTest()));

	// OCT 相關(主要)
	connect(ui.SaveLocationBtn,								SIGNAL(clicked()),				this,	SLOT(ChooseSaveLocaton()));
	connect(ui.AutoSaveSingleRawDataWhileScan_CheckBox,		SIGNAL(stateChanged(int)),		this,	SLOT(AutoSaveWhileScan_ChangeEvent(int)));
	connect(ui.AutoSaveMultiRawDataWhileScan_CheckBox,		SIGNAL(stateChanged(int)),		this,	SLOT(AutoSaveWhileScan_ChangeEvent(int)));
	connect(ui.AutoSaveImageWhileScan_CheckBox,				SIGNAL(stateChanged(int)),		this,	SLOT(AutoSaveWhileScan_ChangeEvent(int)));
	connect(ui.ScanButton,									SIGNAL(clicked()),				this,	SLOT(ScanOCTMode()));

	// OCT 測試
	connect(ui.RawDataToImage,								SIGNAL(clicked()),				this,	SLOT(ReadRawDataToImage()));
	connect(ui.EasyBorderDetect,							SIGNAL(clicked()),				this,	SLOT(ReadRawDataForBorderTest()));
	connect(ui.SingleImageShakeTestButton,					SIGNAL(clicked()),				this,	SLOT(ReadSingleRawDataForShakeTest()));
	connect(ui.MultiImageShakeTestButton,					SIGNAL(clicked()),				this,	SLOT(ReadMultiRawDataForShakeTest()));

	// 點雲操作
	connect(ui.PCIndex,										SIGNAL(currentIndexChanged(int)),this,	SLOT(PCIndexChangeEvnet(int)));
	/*connect(ui.QuaternionWValue,							SIGNAL(editingFinished()),		this,	SLOT(QuaternionChangeEvent()));
	connect(ui.QuaternionXValue,							SIGNAL(editingFinished()),		this,	SLOT(QuaternionChangeEvent()));
	connect(ui.QuaternionYValue,							SIGNAL(editingFinished()),		this,	SLOT(QuaternionChangeEvent()));
	connect(ui.QuaternionZValue,							SIGNAL(editingFinished()),		this,	SLOT(QuaternionChangeEvent()));
	connect(ui.EulerBarX,									SIGNAL(valueChanged(int)),		this,	SLOT(EulerChangeEvent(int)));
	connect(ui.EulerBarY,									SIGNAL(valueChanged(int)),		this,	SLOT(EulerChangeEvent(int)));
	connect(ui.EulerBarZ,									SIGNAL(valueChanged(int)),		this,	SLOT(EulerChangeEvent(int)));*/
	connect(ui.AlignLastTwoPCButton,						SIGNAL(clicked()),				this,	SLOT(AlignLastTwoEvent()));

	// 顯示部分
	connect(ui.ScanNumSlider,								SIGNAL(valueChanged(int)),		this,	SLOT(ScanNumSlider_Change(int)));
	connect(UpdateGLTimer,									SIGNAL(timeout()),				this,	SLOT(DisplayPanelUpdate()));
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
	
	rawManager.bleManager.SendUIPointer(objList);

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
	objList.push_back(ui.QuaternionXValue);
	objList.push_back(ui.QuaternionYValue);
	objList.push_back(ui.QuaternionZValue);
	objList.push_back(ui.QuaternionWValue);
	objList.push_back(ui.EulerBarX);
	objList.push_back(ui.EulerBarY);
	objList.push_back(ui.EulerBarZ);
	objList.push_back(ui.EulerXValueText);
	objList.push_back(ui.EulerYValueText);
	objList.push_back(ui.EulerZValueText);
	objList.push_back(ui.OtherSideResult);

	rawManager.SendUIPointer(objList);

	// 傳送 rawManager 到 OpenGL Widget
	ui.DisplayPanel->SetRawDataManager(&rawManager);
	#pragma endregion
}

// 藍芽事件
void DentistProjectV2::SearchCOM()
{
	QStringList COMListArray = rawManager.bleManager.GetCOMPortsArray();
	ui.COMList->clear();
	ui.COMList->addItems(COMListArray);
}
void DentistProjectV2::ConnectCOM()
{
	rawManager.bleManager.Initalize(ui.COMList->currentText());
}
void DentistProjectV2::ScanBLEDevice()
{
	if (rawManager.bleManager.IsInitialize())
		rawManager.bleManager.Scan();
}
void DentistProjectV2::ConnectBLEDeivce()
{
	rawManager.bleManager.Connect(ui.BLEDeviceList->currentIndex());
}
void DentistProjectV2::SetRotationMode()
{
	if (ui.ResetRotationMode->text() == RotationModeOFF_String)
	{
		ui.DisplayPanel->SetRotationMode(true);
		ui.ResetRotationMode->setText(RotationModeON_String);
		ui.ResetRotationMode->setDefault(true);
	}
	else
	{
		ui.DisplayPanel->SetRotationMode(false);
		ui.ResetRotationMode->setText(RotationModeOFF_String);
		ui.ResetRotationMode->setDefault(false);
	}
}
void DentistProjectV2::GyroResetToZero()
{
	rawManager.bleManager.SetOffsetQuat();
}
void DentistProjectV2::ConnectBLEDevice_OneBtn()
{
	#pragma region 預先設定
	rawManager.bleManager.SetConnectDirectly(BLEDeviceName.toStdString(), BLEDeviceAddress.toStdString());
	#pragma endregion
	#pragma region COM
	QStringList COMListArray = rawManager.bleManager.GetCOMPortsArray();
	ui.COMList->addItems(COMListArray);
	for (int i = 0; i < ExceptCOMName.size(); i++)
		for (int j = 0; j < COMListArray.size(); j++)
			if (ExceptCOMName[i] == COMListArray[j])
			{
				COMListArray.removeAt(j);
				break;
			}
	
	// 例外判斷
	if (COMListArray.size() == 0)
		assert(false && "COM 設定可能有錯誤!!");

	ui.COMList->setCurrentText(COMListArray[0]);
	#pragma endregion
	#pragma region 連結 COM & 自動連結
	rawManager.bleManager.Initalize(ui.COMList->currentText());
	#pragma endregion

}

// 藍芽、九軸測試
void DentistProjectV2::PointCloudAlignmentTest()
{
	QVector<QString> FileInfo;
	QString GyroFileName = QFileDialog::getOpenFileName(this, codec->toUnicode("Gyro 檔案"), "D:/Dentist/Data/ScanData/", "Gyro.txt", nullptr, QFileDialog::DontUseNativeDialog);
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

// OCT 測試
void DentistProjectV2::ReadRawDataToImage()
{
	QString RawFileName = QFileDialog::getOpenFileName(this, codec->toUnicode("RawData 轉圖"), "D:/Dentist/Data/ScanData/", "", nullptr, QFileDialog::DontUseNativeDialog);
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
	QString RawFileName = QFileDialog::getOpenFileName(this, codec->toUnicode("邊界測試"), "D:/Dentist/Data/ScanData/", "", nullptr, QFileDialog::DontUseNativeDialog);
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
	QStringList RawFileName = QFileDialog::getOpenFileNames(this, codec->toUnicode("晃動測式"), "D:/Dentist/Data/ScanData/", "", nullptr, QFileDialog::DontUseNativeDialog);
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

// 點雲操作
void DentistProjectV2::PCIndexChangeEvnet(int)
{
	if (!rawManager.IsLockPC)
	{
		int index = ui.PCIndex->currentIndex();
		//cout << index << endl;
	}
}
void DentistProjectV2::QuaternionChangeEvent()
{

}
void DentistProjectV2::EulerChangeEvent(int)
{

}
void DentistProjectV2::AlignLastTwoEvent()
{
	if (rawManager.PointCloudArray.size() >= 2)
		rawManager.AlignmentPointCloud();
}

// 顯示部分的事件
void DentistProjectV2::ScanNumSlider_Change(int value)
{
	rawManager.ShowImageIndex(value);
	ui.ScanNum_Value->setText(QString::number(value));
}
void DentistProjectV2::DisplayPanelUpdate()
{
	if(!rawManager.IsLockPC)
		ui.DisplayPanel->update();
}