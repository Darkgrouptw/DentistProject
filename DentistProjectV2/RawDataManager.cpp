#include <QOpenGLWidget>				// 因為會跟 OpenCV 3 衝突
#include "OpenGLWidget.h"				// 住要是怕互 Call，Include 會成為無限遞迴
#include "RawDataManager.h"

RawDataManager::RawDataManager()
{
	#pragma region 初始化裝置以外的設定
	cout << "OpenCV Version: " << CV_VERSION << endl;
	DManager.ReadCalibrationData();

	// 設定 Function Pointer
	ScanSingle_Pointer			= bind(&RawDataManager::ScanSingleDataFromDeviceV2,		this, placeholders::_1, placeholders::_2);
	ScanMulti_Pointer			= bind(&RawDataManager::ScanMultiDataFromDeviceV2,		this, placeholders::_1, placeholders::_2);
	TransforImage_Pointer		= bind(&RawDataManager::TransformToIMG,					this, placeholders::_1);
	TransformToOtherSideView_Pointer = bind(&RawDataManager::TransformToOtherSideView,	this);
	CopySingleBorder_Pointer	= bind(&RawDataManager::CopySingleBorder,				this, placeholders::_1);
	ShakeDetect_Single_Pointer	= bind(&RawDataManager::ShakeDetect_Single,				this, placeholders::_1, placeholders::_2);
	ShakeDetect_Multi_Pointer	= bind(&RawDataManager::ShakeDetect_Multi,				this, placeholders::_1, placeholders::_2);
	SavePointCloud_Pointer		= bind(&RawDataManager::SavePointCloud,					this, placeholders::_1);
	AlignmentPointCloud_Pointer = bind(&RawDataManager::AlignmentPointCloud,			this);
	ShowImageIndex_Pointer		= bind(&RawDataManager::ShowImageIndex,					this, 60);

	// 傳進 Scan Thread 中
	Worker = gcnew ScanningWorkerThread(DManager.prop.SizeX);
	Worker->InitScanFunctionPointer(&ScanSingle_Pointer, &ScanMulti_Pointer, &TransforImage_Pointer, &TransformToOtherSideView_Pointer);
	Worker->IntitShakeDetectFunctionPointer(&CopySingleBorder_Pointer, &ShakeDetect_Single_Pointer, &ShakeDetect_Multi_Pointer);
	Worker->InitShowFunctionPointer(&SavePointCloud_Pointer, &AlignmentPointCloud_Pointer, &ShowImageIndex_Pointer);
	#pragma endregion
	#pragma region 初始化裝置
	#ifndef TEST_NO_OCT
	OCT64::OCT64::Init(
		4,
		OCT_DeviceID
	);
	cout << "初始化裝置ID: " << OCT_DeviceID << endl;
	#endif
	#pragma endregion
	#pragma region 創建 Images 資料夾
	vector<QString>testDir = { "origin_v2", "border_v2" };
	QDir dir(".");

	for (int i = 0; i < testDir.size(); i++)
		dir.mkpath("Images/OCTImages/" + testDir[i]);
	#pragma endregion
}
RawDataManager::~RawDataManager()
{
	// 刪除 tempDir
	tempDir.remove();
	cout << "刪除暫存!!" << endl;
}

// UI 相關
void RawDataManager::SendUIPointer(QVector<QObject*> UIPointer)
{
	// 確認是不是有多傳，忘了改的
	assert(UIPointer.size() == 10 && "UI 對應有問題!!");
	ImageResult				= (QLabel*)UIPointer[0];
	BorderDetectionResult	= (QLabel*)UIPointer[1];
	NetworkResult			= (QLabel*)UIPointer[2];
	OtherSideResult			= (QLabel*)UIPointer[8];
	NetworkOtherSide		= (QLabel*)UIPointer[9];

	// 後面兩個是 給 ScanThread
	QSlider* slider			= (QSlider*)UIPointer[3];
	QPushButton* scanButton = (QPushButton*)UIPointer[4];
	QLineEdit* savePathText	= (QLineEdit*)UIPointer[5];
	
	// OpenGL
	DisplayPanel			= UIPointer[6];

	// 點雲顯示
	PCIndex					= (QComboBox*)UIPointer[7];

	Worker->InitUIPointer(slider, scanButton, savePathText);
}
void RawDataManager::ShowImageIndex(int index)
{
	if (60 <= index && index <= 200)
	{
		if (QImageResultArray.size() > 1)				
		{
			// Multi
			QImage Pixmap_ImageResult = QImageResultArray[index];
			ImageResult->setPixmap(QPixmap::fromImage(Pixmap_ImageResult));

			QImage Pixmap_BorderDetectionResult = QBorderDetectionResultArray[index];
			BorderDetectionResult->setPixmap(QPixmap::fromImage(Pixmap_BorderDetectionResult));

			if (QNetworkResultArray.size() > 0) 
			{
				QImage Pixmap_NetworkResult = QNetworkResultArray[index - 60];
				NetworkResult->setPixmap(QPixmap::fromImage(Pixmap_NetworkResult));
			}
		}
		else if (QImageResultArray.size() == 1)
		{
			// Single
			QImage Pixmap_ImageResult = QImageResultArray[0];
			ImageResult->setPixmap(QPixmap::fromImage(Pixmap_ImageResult));

			QImage Pixmap_BorderDetectionResult = QBorderDetectionResultArray[0];
			BorderDetectionResult->setPixmap(QPixmap::fromImage(Pixmap_BorderDetectionResult));
		}
	}	
}

// OCT 相關的步驟
RawDataType RawDataManager::ReadRawDataFromFileV2(QString FileName)
{
	// 起始
	clock_t startT, endT;
	startT = clock();

	QFile inputFile(FileName);
	if (!inputFile.open(QIODevice::ReadOnly))
	{
 		cout << "Raw Data 讀取錯誤" << endl;
		return RawDataType::ERROR_TYPE;
	}
	else
		cout << "讀取 Raw Data: " << FileName.toLocal8Bit().toStdString() << endl;

	int bufferSize = inputFile.size() / sizeof(quint8);

	QDataStream qData(&inputFile);
	QByteArray buffer;
	buffer.clear();
	buffer.resize(bufferSize);
	qData.readRawData(buffer.data(), bufferSize);

	inputFile.close();

	// 結算
	endT = clock();
	cout << "讀取時間: " << (endT - startT) / (double)(CLOCKS_PER_SEC) << " sec" << endl;

	RawDataProperty prop = DManager.prop;
	if (bufferSize <= prop.SizeX * prop.SizeZ * 8)
	{
		//RawDataManager
		cout << "讀取單張 RawData !!" << endl;
		cudaV2.SingleRawDataToPointCloud(
			buffer.data(), bufferSize,
			prop.SizeX, prop.SizeZ,
			prop.ShiftValue, prop.K_Step, prop.CutValue);
		return RawDataType::SINGLE_DATA_TYPE;
	}
	else
	{
		//RawDataManager
		cout << "讀取多張 RawData !!" << endl;
		cudaV2.MultiRawDataToPointCloud(
			buffer.data(), bufferSize,
			prop.SizeX, prop.SizeY, prop.SizeZ,
			prop.ShiftValue, prop.K_Step, prop.CutValue);
		return RawDataType::MULTI_DATA_TYPE;
	}
}
void RawDataManager::ScanSingleDataFromDeviceV2(QString SaveFileName, bool NeedSave_RawData)
{
	#pragma region 開 Port
	SerialPort port(gcnew System::String(OCTDevicePort.c_str()), 9600);
	port.Open();

	// 是否開慢軸
	port.RtsEnable = false;

	// 如果沒有開成功，或是搶 Port 會報錯
	if (!port.IsOpen)
	{
		cout << "OCT 的 COM 打不開!!" << endl;
		return;
	}
	#pragma endregion
	#pragma region 開始掃描
	float LV_65 = 65;
	unsigned int SampleCount = 2048;
	OCT_DataLen = 1;
	OCT_ErrorBoolean = false;
	System::String^ ErrorString = gcnew System::String("");
	string ErrorCString = "";
	System::String^ SaveFileName_C = gcnew System::String(SaveFileName.toStdString().c_str());

	OCT64::OCT64::StartCap(
		OCT_DeviceID,						// 裝置 ID
		OCT_HandleOut,						// Handle (要傳給 Scan 的)
		LV_65,								// 直流
		SampleCount,						// 2048
		OCT_DataLen,						// 資料長度
		NeedSave_RawData,					// 這個好像是要步要 output
		SaveFileName_C,						// 儲存位置
		OCT_ErrorBoolean,					// 是否要有 Error
		ErrorString							// 錯誤訊息
	);

	// 代表硬體有問題
	ErrorCString = MarshalString(ErrorString);
	if (ErrorCString != OCT_SUCCESS_TEXT.toStdString())
	{
		// 先關閉
		OCT64::OCT64::AboutADC(OCT_DeviceID);
		port.Close();

		cout << "OCT StartCap Error String: " << ErrorCString << endl;
		assert(false && "OCT Start 有 Bug!!");
	}

	// 要接的 Array
	cli::array<unsigned short>^ OutDataArray = gcnew cli::array<unsigned short>(OCT_PIC_SIZE);				// 暫存的 Array
	unsigned short* Final_OCT_Array = new unsigned short[OCT_PIC_SIZE];										// 取值得 Array
	char* Final_OCT_Char = new char[OCT_PIC_SIZE * 2];														// 最後要丟到 Cuda 的 Array

	// 動慢軸
	OCT_AllDataLen = OCT_DataLen * 2;

	// 掃描
	cli::array<unsigned short>^ inputArray = gcnew cli::array<unsigned short>(OCT_PIC_SIZE);
	OCT64::OCT64::Scan(
		OCT_HandleOut,					// Handle
		OCT_AllDataLen,					// 資料大小
		inputArray,						// 這個沒有用
		OutDataArray,					// 掃描的結果
		ErrorString						// 錯誤訊息
	);

	// 代表硬體有問題
	ErrorCString = MarshalString(ErrorString);
	if (ErrorCString != OCT_SUCCESS_TEXT.toStdString())
	{
		// 先關閉
		OCT64::OCT64::AboutADC(OCT_DeviceID);
		port.Close();

		cout << "Scan Error String: " << ErrorCString << endl;
		assert(false && "OCT Scan 有 Bug!!");
	}

	// cli Array 轉到 manage array
	pin_ptr<unsigned short> pinPtrArray = &OutDataArray[OutDataArray->GetLowerBound(0)];
	memcpy(Final_OCT_Array, pinPtrArray, sizeof(unsigned short) * OCT_PIC_SIZE);
	#pragma endregion
	#pragma region 關閉已開的 Port
	OCT64::OCT64::AboutADC(OCT_DeviceID);
	port.Close();
	#pragma endregion
	#pragma region 丟到 Cuda 上解資料
	// 要先轉成 Char
	OCT_DataType_Transfrom(Final_OCT_Array, OCT_PIC_SIZE, Final_OCT_Char);

	// 要將資料轉到陣列上
	RawDataProperty prop = DManager.prop;
	cudaV2.SingleRawDataToPointCloud(
		Final_OCT_Char, OCT_PIC_SIZE * 2,
		prop.SizeX, prop.SizeZ,
		prop.ShiftValue, prop.K_Step, prop.CutValue
	);
	#pragma endregion
	#pragma region 刪除 New 出來的 Array
	delete Final_OCT_Array;
	delete Final_OCT_Char;
	#pragma endregion
}
void RawDataManager::ScanMultiDataFromDeviceV2(QString SaveFileName, bool NeedSave_RawData)
{
	#pragma region 開 Port
	SerialPort port(gcnew System::String(OCTDevicePort.c_str()), 9600);
	port.Open();

	// 是否開慢軸
	port.RtsEnable = true;

	// 如果沒有開成功，或是搶 Port 會報錯
	if (!port.IsOpen)
	{
		cout << "OCT 的 COM 打不開!!" << endl;
		return;
	}
	#pragma endregion
	#pragma region 開始掃描
	float LV_65 = 65;
	unsigned int SampleCount = 2048;
	OCT_DataLen = 1;
	OCT_ErrorBoolean = false;
	System::String^ ErrorString = gcnew System::String("");
	string ErrorCString = "";
	System::String^ SaveFileName_C = gcnew System::String(SaveFileName.toStdString().c_str());

	OCT64::OCT64::StartCap(
		OCT_DeviceID,						// 裝置 ID
		OCT_HandleOut,						// Handle (要傳給 Scan 的)
		LV_65,								// 直流
		SampleCount,						// 2048
		OCT_DataLen,						// 資料長度
		NeedSave_RawData,					// 這個好像是要步要 output
		SaveFileName_C,						// 儲存位置
		OCT_ErrorBoolean,					// 是否要有 Error
		ErrorString							// 錯誤訊息
	);

	// 代表硬體有問題
	ErrorCString = MarshalString(ErrorString);
	if (ErrorCString != OCT_SUCCESS_TEXT.toStdString())
	{
		// 先關閉
		OCT64::OCT64::AboutADC(OCT_DeviceID);
		port.Close();

		cout << "OCT StartCap Error String: " << ErrorCString << endl;
		assert(false && "OCT Start 有 Bug!!");
	}

	// 要接的 Array
	cli::array<unsigned short>^ OutDataArray = gcnew cli::array<unsigned short>(OCT_PIC_SIZE);				// 暫存的 Array
	unsigned short* Final_OCT_Array = new unsigned short[OCT_PIC_SIZE * 125];								// 取值得 Array
	unsigned short* Temp_OCT_Pointer = Final_OCT_Array;														// 暫存，因為會一直位移
	char* Final_OCT_Char = new char[OCT_PIC_SIZE * 250];													// 最後要丟到 Cuda 的 Array

	// 動慢軸
	OCT_AllDataLen = OCT_DataLen * 2;
	int PicNumber = 0;
	while (PicNumber < 125) {
		// 掃描
		cli::array<unsigned short>^ inputArray = gcnew cli::array<unsigned short>(OCT_PIC_SIZE);
		OCT64::OCT64::Scan(
			OCT_HandleOut,					// Handle
			OCT_AllDataLen,					// 資料大小
			inputArray,						// 這個沒有用
			OutDataArray,					// 掃描的結果
			ErrorString						// 錯誤訊息
		);

		// 代表硬體有問題
		ErrorCString = MarshalString(ErrorString);
		if (ErrorCString != OCT_SUCCESS_TEXT.toStdString())
		{
			// 先關閉
			OCT64::OCT64::AboutADC(OCT_DeviceID);
			port.Close();

			cout << "Scan Error String: " << ErrorCString << endl;
			assert(false && "OCT Scan 有 Bug!!");
		}

		// cli Array 轉到 manage array
		pin_ptr<unsigned short> pinPtrArray = &OutDataArray[OutDataArray->GetLowerBound(0)];
		memcpy(Temp_OCT_Pointer, pinPtrArray, sizeof(unsigned short) * OCT_PIC_SIZE);
		Temp_OCT_Pointer += OCT_PIC_SIZE;

		// 繼續往下掃
		PicNumber++;
	}
	#pragma endregion
	#pragma region 關閉已開的 Port
	OCT64::OCT64::AboutADC(OCT_DeviceID);
	port.RtsEnable = false;
	port.Close();
	#pragma endregion
	#pragma region 丟到 Cuda 上解資料
	// 要先轉成 Char
	OCT_DataType_Transfrom(Final_OCT_Array, OCT_PIC_SIZE * 125, Final_OCT_Char);

	// 要將資料轉到 DManager.rawDP 上
	RawDataProperty prop = DManager.prop;
	cudaV2.MultiRawDataToPointCloud(
		Final_OCT_Char, OCT_PIC_SIZE * 250,
		prop.SizeX, prop.SizeY, prop.SizeZ,
		prop.ShiftValue, prop.K_Step, prop.CutValue
	);
	// cudaV2.RawDataToPointCloud(buffer.data(), bufferSize, 250, 250, 2048, 37 * 4 - 4, 2, 10);
	#pragma endregion
	#pragma region 刪除 New 出來的 Array
	delete Final_OCT_Array;
	delete Final_OCT_Char;
	#pragma endregion
}
void RawDataManager::TransformToIMG(bool NeedSave_Image = false)
{
	#pragma region 開始時間
	#ifdef SHOW_TRCUDAV2_TRANSFORM_TIME
	clock_t startT, endT;
	startT = clock();
	#endif
	#pragma endregion
	#pragma region 清空其他 Array
	// 如果跑出結果是全黑的，那有可能是顯卡記憶體不夠的問題
	ImageResultArray.clear();
	BorderDetectionResultArray.clear();

	QImageResultArray.clear();
	QBorderDetectionResultArray.clear();

	NetworkResultArray.clear();
	QNetworkResultArray.clear();
	#pragma endregion
	#pragma region 塞進 Array
	vector<Mat> TempMatArray;

	// 轉換到 Vector 中
	TempMatArray = cudaV2.TransfromMatArray(false);
	ImageResultArray = QVector<Mat>::fromStdVector(TempMatArray);

	TempMatArray = cudaV2.TransfromMatArray(true);
	BorderDetectionResultArray = QVector<Mat>::fromStdVector(TempMatArray);

	// 轉 QImage
	for (int i = 0; i < ImageResultArray.size(); i++)
	{
		QImage tempQImage = Mat2QImage(ImageResultArray[i], CV_8UC3);
		QImageResultArray.push_back(tempQImage);

		tempQImage = Mat2QImage(BorderDetectionResultArray[i], CV_8UC3);
		QBorderDetectionResultArray.push_back(tempQImage);

		if (NeedSave_Image)
		{
			// 原圖
			cv::imwrite("Images/OCTImages/origin_v2/" + to_string(i) + ".png", ImageResultArray[i]);
			
			// Combine 結果圖
			cv::imwrite("Images/OCTImages/border_v2/" + to_string(i) + ".png", BorderDetectionResultArray[i]);
		}
	}
	#pragma endregion
	#pragma region 結束時間
	#ifdef SHOW_TRCUDAV2_TRANSFORM_TIME
	endT = clock();

	if (NeedSave_Image)
		cout << "有存出圖片";
	else
		cout << "無存出圖片";
	cout << "，轉圖檔完成: " << (endT - startT) / (double)(CLOCKS_PER_SEC) << "s" << endl;
	#endif
	#pragma endregion
}
void RawDataManager::TransformToOtherSideView()
{
	assert(ImageResultArray.size() > 1 && "呼叫此函式必須要有多張圖才可以呼叫");
	#pragma region 開始時間
	#ifdef SHOW_TRCUDAV2_TRANSFORM_TIME
	clock_t startT, endT;
	startT = clock();
	#endif
	#pragma endregion
	#pragma region TopView
	Mat result = cudaV2.TransformToOtherSideView();
	QImage qresult = Mat2QImage(result, CV_8UC3);
	OtherSideResult->setPixmap(QPixmap::fromImage(qresult));

	// 順便存一分到 外圍
	cvtColor(result, OtherSideMat, CV_BGR2GRAY);
	#pragma endregion
	#pragma region 結束時間
	#ifdef SHOW_TRCUDAV2_TRANSFORM_TIME
	endT = clock();
	cout << "TopView 轉換時間: " << (endT - startT) / (double)(CLOCKS_PER_SEC) << "s" << endl;
	#endif
	#pragma endregion
}
void RawDataManager::SetScanOCTMode(bool IsStart, QString* EndText, bool NeedSave_Single_RawData, bool NeedSave_Multi_RawData, bool NeedSave_ImageData, bool AutoDelete_ShakeData)
{
	// 如果開始的話，就清空資料
	if (IsStart)
	{
		PointCloudArray.clear();
		PCWidgetUpdate();
	}

	// 設定
	Worker->SetParams(EndText, NeedSave_Single_RawData, NeedSave_Multi_RawData, NeedSave_ImageData, AutoDelete_ShakeData);
	Worker->SetScanMode(IsStart);
}
void RawDataManager::SetScanOCTOnceMode()
{
	Worker->SetScanOnceMode();
}
void RawDataManager::CopySingleBorder(int *&LastData_Pointer)
{
	// 如果是空的，就給一段位置
	if (LastData_Pointer == NULL)
		LastData_Pointer = new int[DManager.prop.SizeX];
	cudaV2.CopySingleBorder(LastData_Pointer);
}
bool RawDataManager::ShakeDetect_Single(int* LastData, bool ShowDebugMessage)
{
	return cudaV2.ShakeDetect_Single(LastData, ShowDebugMessage);
}
bool RawDataManager::ShakeDetect_Multi(bool UsePreciseThreshold, bool ShowDebugMessage)
{
	return cudaV2.ShakeDetect_Multi(UsePreciseThreshold, ShowDebugMessage);
}
void RawDataManager::SavePointCloud(QQuaternion quat)
{
	#pragma region 創建 Array
	// 邊界 & 其他資訊
	RawDataProperty prop = DManager.prop;
	int* BorderData = new int[prop.SizeX *DManager.prop.SizeY];
	cudaV2.CopyBorder(BorderData);

	// 點雲
	PointCloudInfo info;
	#pragma endregion
	#pragma region 轉成點雲
	// 產生 Rotation Matrix;
	QMatrix4x4 rotationMatrix;
	rotationMatrix.setToIdentity();
	rotationMatrix.rotate(quat);

	QVector3D MidPoint;
	for (int y = 0; y < prop.SizeY; y++)
		for (int x = 0; x < prop.SizeX; x++)
		{
			int index = y * prop.SizeX + x;			// 對應到 Border Data
			int MapID = (y * prop.SizeX + x) * 2;	// 對應到 Mapping Matrix，在讀取的時候他是兩筆為一個單位
			if (BorderData[index] != -1)
			{
				// 轉到 3d 座標
				QVector3D pointInSpace;
				pointInSpace.setX(DManager.MappingMatrix[MapID + 0]);
				pointInSpace.setY(DManager.MappingMatrix[MapID + 1]);
				pointInSpace.setZ(BorderData[index] * DManager.zRatio / prop.SizeZ * 2);

				MidPoint += pointInSpace;


				// 加進 Point 陣列裡
				info.Points.push_back(pointInSpace);
			}
		}

	// 如果讀到是空的，就跳過
	if (info.Points.size() == 0)
	{
		delete[] BorderData;
		return;
	}

	// 中心的點 & 加入九軸資訊
	MidPoint /= info.Points.size();
	for (int i = 0; i < info.Points.size(); i++)
		info.Points[i] = (rotationMatrix * QVector4D(info.Points[i] - MidPoint, 1)).toVector3D() + QVector3D(0, MidPoint.y(), 0);

	// 加進陣列裡
	PointCloudArray.push_back(info);

	#pragma endregion 
	#pragma region 刪除 Array
	delete[] BorderData;
	#pragma endregion
	#pragma region PC Index
	// 先重新設定 PCIndex
	QuaternionList.push_back(quat);

	// 讓他往前
	if (SelectIndex == PointCloudArray.size() - 2)
		SelectIndex++;

	// 需更新
	PCWidgetUpdate();
	#pragma endregion
}
void RawDataManager::AlignmentPointCloud()
{
	// 點雲拼接
	if (PointCloudArray.size() > 1)
	{
		#pragma region 先將原本的點，先轉至正確位置
		int LastID = PointCloudArray.size() - 1;

		////增加新進點雲Matrix
		if (InitRotationMarix.size() != PointCloudArray.size()) 
			InitRotationMarix.push_back(QMatrix4x4());

		////計算要轉至哪個Matrix
		QMatrix4x4 LastRotationMatrix;
		for (int i = 0; i < PointCloudArray.size(); i++)
			LastRotationMatrix = InitRotationMarix[i] * LastRotationMatrix;

		////如果是新Matrix表示要更新點雲位置
		if (InitRotationMarix[LastID].isIdentity())
			for (int i = 0; i < PointCloudArray[LastID].Points.size(); i++)
			{
				QVector4D point4D = QVector4D(PointCloudArray[LastID].Points[i], 1);
				QVector3D pointToRightPlace = (LastRotationMatrix * point4D).toVector3D();		// 轉到正確的位置
				PointCloudArray[LastID].Points[i] = pointToRightPlace;
			}

		#pragma endregion
		#pragma region 拿最後一篇跟其他拼接
		vector<Point3D> NewPC, LastPC;
		ConvertQVector2Point3D(PointCloudArray[LastID].Points, NewPC);
		ConvertQVector2Point3D(PointCloudArray[LastID - 1].Points, LastPC);

		// 轉換 Matrix
		float score = 0;
		QMatrix4x4 rotationMatrix = super4PCS_Align(&LastPC, &NewPC, score).transposed();

		InitRotationMarix[LastID] = rotationMatrix * InitRotationMarix[LastID];

		cout << "拼接最後分數: " << score << endl;
		ConvertPoint3D2QVector(NewPC, PointCloudArray[LastID].Points);
		#pragma endregion
		#pragma region 判斷否要保留
		// 這邊再去做判斷
		// 如果分數小於一個 Threshold 那就丟掉
		if (score < AlignScoreThreshold)
		{
			//// 如果丟掉移除最後一個矩陣
			InitRotationMarix.removeLast();

			PointCloudArray.removeLast();
			SelectIndex = PointCloudArray.size() - 1;
			PCWidgetUpdate();
		}
		else
		{
			IsLockPC = true;	// 要重新更新點雲了
		}
		#pragma endregion
	}
	else {
		////第一片是單位矩陣
		InitRotationMarix.push_back(QMatrix4x4());
	}
}
void RawDataManager::CombinePointCloud(int FirstID, int LastID)
{
	// 加進去
	for (int i = 0; i < PointCloudArray[LastID].Points.size(); i++)
	{
		QVector3D p = PointCloudArray[LastID].Points[i];
		PointCloudArray[FirstID].Points.append(p);
	}
	InitRotationMarix[FirstID] = InitRotationMarix[LastID];

	// 把顯示部分設定為合併後的那一個
	SelectIndex = FirstID;

	// 刪除那片點雲
	PointCloudArray.removeAt(LastID);
	InitRotationMarix.removeAt(LastID);

	// 更新資料
	PCWidgetUpdate();
}

// Network or Volume 相關的 Function
void RawDataManager::NetworkDataGenerateV2(QString rawDataPath)
{
	RawDataType t = ReadRawDataFromFileV2(rawDataPath);
	if (RawDataType::MULTI_DATA_TYPE == t)
	{
		// 轉成圖片並儲存
		TransformToIMG(true);

		// 抓取 Bounding Box
		QFile BoundingBoxFile("./Images/OCTImages/boundingBox.txt");
		BoundingBoxFile.open(QIODevice::WriteOnly);
		QTextStream ss(&BoundingBoxFile);

		ss << "TopLeft (x, y), ButtomRight (x, y)" << endl;
		for (int i = 0; i < ImageResultArray.size(); i++)
		{
			QVector2D topLeft, buttomRight;
			Mat img = GetBoundingBox(ImageResultArray[i], topLeft, buttomRight);
			imwrite("./Images/OCTImages/bounding_v2/" + to_string(i) + ".png", img);

			ss << topLeft.x() << " " << topLeft.y() << " " << buttomRight.x() << " " << buttomRight.y() << endl;
		}
		BoundingBoxFile.close();

		// Top View
		Mat result = cudaV2.TransformToOtherSideView();
		Mat GrayResult;
		cvtColor(result, GrayResult, CV_BGR2GRAY);
		cv::imwrite("Images/OCTImages/OtherSide.png", GrayResult);
		OtherSideMat = GrayResult.clone();

		QImage qreulst = Mat2QImage(result, CV_8UC3);
		OtherSideResult->setPixmap(QPixmap::fromImage(qreulst));
		cout << "儲存完成!!" << endl;
	}
	else
		cout << "不能使用單層資料的資料!!" << endl;
}
void RawDataManager::NetworkDataGenerateInRamV2()
{
	#pragma region 例外判斷
	if (ImageResultArray.size() != 250)
	{
		cout << "資料必須是立體資料!!" << endl;
		return;
	}
	#pragma endregion
	#pragma region 產生 Bounding Box
	// 更新點雲
	TLPointArray.clear();
	BRPointArray.clear();

	// 整個 Bounding Box Parse 過去
	for (int i = 0; i < ImageResultArray.size(); i++)
	{
		QVector2D topLeft, buttomRight;
		GetBoundingBox(ImageResultArray[i], topLeft, buttomRight);
		TLPointArray.push_back(topLeft);
		BRPointArray.push_back(buttomRight);
	}
	#pragma endregion
}
bool RawDataManager::CheckIsValidData()
{
	if (TLPointArray.size() != 250 || BRPointArray.size() != 250)
		return false;

	QVector2D BoundingBox = BRPointArray[60] - TLPointArray[60];
	if (BoundingBox.x() < 150 || BoundingBox.y() < 150)
		return false;
	return true;
}

void RawDataManager::PredictOtherSide()
{
	assert(!OtherSideMat.empty() && "不能為空的!!");
	cout << "Temp Dir: " << tempDir.path().toLocal8Bit().toStdString() << endl;
	if (tempDir.isValid())
	{
		QString tensorflowProcessFilePath = "./DentistProjectV2_TensorflowNetProcess.exe";
		if (QFile::exists(tensorflowProcessFilePath))
		{
			// 寫出檔案
			QProcess tensorflowProcess;
			QString tempImgPath = tempDir.filePath("OtherSide.png");
			QString tempPredictImgPath = tempDir.filePath("Predict.png");

			// 塞參數
			QStringList params;
			params.append(tempImgPath);
			params.append(tempPredictImgPath);
			params.append("OtherSide");				// Mode: OtherSide
			cv::imwrite(tempImgPath.toLocal8Bit().toStdString(), OtherSideMat);

			// 開始 Process
			QElapsedTimer counterTimer;
			counterTimer.start();
			//tensorflowProcess.setProcessChannelMode(QProcess::MergedChannels);
			tensorflowProcess.start(tensorflowProcessFilePath, params);
			if (tensorflowProcess.waitForFinished())
			{
				cout << "======================================================" << endl;
				cout << "Process Output: " << endl << endl;
				cout << tensorflowProcess.readAllStandardOutput().toStdString() << endl;
				#ifndef DISABLE_TENSORFLOW_ERROR_DEBUG
				cout << "Process Error: " << endl << endl;
				cout << tensorflowProcess.readAllStandardError().toStdString() << endl;
				#endif
				cout << "======================================================" << endl;
				cout << "時間: " << (double)(counterTimer.elapsed() / 1000) << " sec" << endl;
			}
			else
			{
				cout << "Process Timeout!!" << endl;
				return;
			}

			Mat PredictOtherSide = imread(tempPredictImgPath.toLocal8Bit().toStdString(), CV_LOAD_IMAGE_GRAYSCALE);
			cvtColor(PredictOtherSide, PredictOtherSide, CV_GRAY2BGR);
			QImage predictQImage = Mat2QImage(PredictOtherSide, CV_8UC3);
			NetworkOtherSide->setPixmap(QPixmap::fromImage(predictQImage));

			// 刪除其他圖片
			//QFile::remove(tempImgPath);
			//QFile::remove(tempPredictImgPath);
		}
		else
			assert(false && "確定要先編過 TensorflowNet Process!!");
	}
	else
		assert(false && "確定站存資料夾已經創立!!");
}
void RawDataManager::PredictFull()
{
	assert(ImageResultArray.size() == 250 && "必須要有資料!!");
	if (tempDir.isValid())
	{
		QString tensorflowProcessFilePath = "./DentistProjectV2_TensorflowNetProcess.exe";
		if (QFile::exists(tensorflowProcessFilePath))
		{
			// 寫出檔案
			QProcess tensorflowProcess;

			// 塞參數
			QStringList params;
			params.append(tempDir.path());
			params.append("60,200");
			params.append("Full");				// Mode: Full

			// 寫出檔案
			cout << "存出檔案結果!!" << endl;
			for (int i = 60; i <= 200; i++)
			{
				QVector2D TL = TLPointArray[i];
				QVector2D BR = BRPointArray[i];
				int width = BR[0] - TL[0];
				int height = BR[1] - TL[1];
				cv::imwrite(tempDir.filePath(QString::number(i) + ".png").toLocal8Bit().toStdString(),
					cv::Mat(ImageResultArray[i], cv::Rect(TL[0], TL[1], width, height)));
			}

			// 開始 Process
			QElapsedTimer counterTimer;
			counterTimer.start();
			tensorflowProcess.start(tensorflowProcessFilePath, params);
			if (tensorflowProcess.waitForFinished(-1))
			{
				cout << "======================================================" << endl;
				cout << "Process Output: " << endl << endl;
				cout << tensorflowProcess.readAllStandardOutput().toStdString() << endl;
				#ifndef DISABLE_TENSORFLOW_ERROR_DEBUG
				cout << "Process Error: " << endl << endl;
				cout << tensorflowProcess.readAllStandardError().toStdString() << endl;
				#endif
				cout << "======================================================" << endl;
				cout << "時間: " << (double)(counterTimer.elapsed() / 1000) << " sec" << endl;
			}
			else
			{
				cout << "Process Timeout!!" << endl;
				return;
			}

		}
		else
			assert(false && "確定要先編過 TensorflowNet Process!!");
	}
	else
		assert(false && "確定站存資料夾已經創立!!");
}
void RawDataManager::LoadPredictImage() 
{
	QString testPath = "E:/DentistData/DentistProjectV2-p3dLon";
	if (tempDir.isValid())
		for (int i = 60; i <= 200; i++)
		{
			cv::Mat BlankImg = cv::Mat(ImageResultArray[0].size(), CV_8UC3);
			cv::Mat LoadImage = cv::imread((testPath + "/Result_" + QString::number(i) + ".png").toLocal8Bit().toStdString(), CV_LOAD_IMAGE_COLOR);

			QVector2D TL = TLPointArray[i];
			QVector2D BR = BRPointArray[i];
			int width = BR[0] - TL[0];
			int height = BR[1] - TL[1];

			LoadImage.copyTo(BlankImg(cv::Rect(TL[0], TL[1], width, height)));
			NetworkResultArray.push_back(BlankImg);
		}
}
void RawDataManager::SmoothNetworkData()
{
	#pragma region 找出最大最小值
	// r => Image rows
	// y => 張數
	// c => Image cols
	// 0, 0 => 是圖片的左上角
	int rMin = INT_MAX, rMax = 0,
		yMin = 60, yMax = 200,
		cMin = INT_MAX, cMax = 0;

	assert(NetworkResultArray.size() == (200 - 60 + 1) && "必須要有 141 張圖!!");
	for (int i = 0; i < NetworkResultArray.size(); i++)
	{
		int index = i + 60;					// Offset 60 張圖

		// 取出點
		QVector2D TL = TLPointArray[i];
		QVector2D BR = BRPointArray[i];

		if (cMin > TL.x()) cMin = TL.x();
		if (rMin > TL.y()) rMin = TL.y();
		if (cMax < BR.x()) cMax = BR.x();
		if (rMax < BR.y()) rMax = BR.y();
	}

	// Clamp 到結果之間
	cMax = clamp(cMax, 0, DManager.prop.SizeZ - 1);
	rMax = clamp(rMax, 0, DManager.prop.SizeX - 1);
	
	cout << "Row: " << rMin << " " << rMax << " Col Max: " << cMin << " " << cMax << endl;
	#pragma endregion
	#pragma region 創建 Table 矩陣
	int yCount = (yMax - yMin) + 1;
	int rCount = (rMax - rMin) + 1;
	int cCount = (cMax - cMin) + 1;

	// [  ] [  ] [  ] [  ]
	// 張數 Rows Cols 種類(0 ~ 4)
	int* __TotalSumAreaTable = new int[yCount * rCount * cCount];
	int** _TotalSumAreaTable = new int*[yCount * rCount];
	int*** TotalSumAreaTable = new int**[yCount];
	memset(__TotalSumAreaTable, 0, sizeof(int) * yCount * rCount * cCount);

	// 傳位置
	for (int i = 0; i < yCount; i++)
	{
		for (int j = 0; j < rCount; j++)
		{
			int Dim2offsetIndex = i * rCount +						// 張數
								j;									// Row
			_TotalSumAreaTable[Dim2offsetIndex] = &__TotalSumAreaTable[Dim2offsetIndex * cCount];
		}
		int Dim1offsetIndex = i;									// 張數
		TotalSumAreaTable[Dim1offsetIndex] = &_TotalSumAreaTable[Dim1offsetIndex * rCount];
	}
	#pragma endregion
	#pragma region 跑每一個點去拿結果
	int WindowSize = 11;
	#pragma omp parallel for 
	for (int i = 0; i < yCount; i++)
		for (int j = 0; j < rCount; j++)
			for (int k = 0; k < cCount; k++)
			{
				// 跑過每一個點把結果加起來平均
				int TotalCount = 0;
				QVector<float> CountArray = { 0,0,0,0 };		// 背景、牙齒、牙齦、齒槽骨
				int halfSize = (WindowSize - 1) / 2;
				for (int ii = -halfSize; ii <= halfSize; ii++)
					for (int jj = -halfSize; jj <= halfSize; jj++)
						for (int kk = -halfSize; kk <= halfSize; kk++)
						{
							// 判斷有沒有在範圍內，如果沒有在範圍內，就跳出
							if ((i + ii) < 0		|| (rMin + j + jj) < 0			|| (cMin + k + kk) < 0 ||
								(i + ii) >= yCount	|| (rMin + j + jj) >= rCount	|| (cMin + k + kk) >= cCount)
								continue;

							Vec3b color = NetworkResultArray[i + ii].at<Vec3b>(rMin + j + jj, cMin + k + kk);		// B G R
							if (color[0] == 255 && color[1] == 0 && color[2] == 0)
								CountArray[3]++;
							else if (color[0] == 0 && color[1] == 255 && color[2] == 0)
								CountArray[2]++;
							else if (color[0] == 0 && color[1] == 0 && color[2] == 255)
								CountArray[1]++;
							else if (color[0] == 0 && color[1] == 0 && color[2] == 0)
								CountArray[0]++;
							TotalCount++;
						}

				// 呈上 Weight
				CountArray[0] *= CONST_BG_WEIGHT;
				CountArray[1] *= CONST_TEETH_WEIGHT;
				CountArray[2] *= CONST_MEET_WEIGHT;
				CountArray[3] *= CONST_BONE_WEIGHT;

				float* max_ele = std::max_element(CountArray.begin(), CountArray.end());
				int MaxIndex = max_ele - CountArray.begin(); 
				TotalSumAreaTable[i][j][k] = MaxIndex;
			}
	#pragma endregion
	#pragma region Smooth 平面
	#pragma omp parallel for 
	for (int i = 0; i < yCount; i++)
		for (int j = 0; j < rCount; j++)
			for (int k = 0; k < cCount; k++)
			{
				// 抓出來
				int MaxIndex = TotalSumAreaTable[i][j][k];

				Vec3b changeColor = Vec3b(0, 0, 0);
				if (MaxIndex == 1) changeColor = Vec3b(0, 0, 255);
				if (MaxIndex == 2) changeColor = Vec3b(0, 255, 0);
				if (MaxIndex == 3) changeColor = Vec3b(255, 0, 0);

				// 貼上
				NetworkResultArray[i].at<Vec3b>(rMin + j, cMin + k) = changeColor;
			}

	// 清除記憶體
	delete[] __TotalSumAreaTable;
	delete[] _TotalSumAreaTable;
	delete[] TotalSumAreaTable;
	#pragma endregion

}
void RawDataManager::NetworkDataToQImage()
{
	for (int i = 0; i < NetworkResultArray.size(); i++)
	{
		QImage qimg = Mat2QImage(NetworkResultArray[i], CV_8UC3);
		QNetworkResultArray.append(qimg);
	}
}

//void RawDataManager::ImportVolumeDataTest(QString boundingBoxPath)
//{
//	//assert 
//	/*int SizeX = DManager.prop.SizeX;
//	int SizeZ = DManager.prop.SizeZ / 2;
//
//	VolumeRenderClass *voxel = new VolumeRenderClass(SizeX, SizeZ, DManager.MappingMatrix, DManager.zRatio);
//	voxel->ImportData(boundingBoxPath);
//	VolumeDataArray.push_back(voxel);
//	IsLockVolumeData = true;*/
//}

// 點雲相關
void RawDataManager::PCWidgetUpdate()
{
	// 更新介面
	IsLockPC = true;
	IsWidgetUpdate = true;

	PCIndex->clear();
	for (int i = 0; i < PointCloudArray.size(); i++)
	{
		QString tempText = QString::number(i) + " (" + QString::number(PointCloudArray[i].Points.size()) + ")";
		PCIndex->addItem(tempText);
	}

	// 這邊是做例外判斷
	if (PointCloudArray.size() <= 0)
		return;

	PCIndex->setCurrentIndex(SelectIndex);

	// 取消
	IsWidgetUpdate = false;
}
void RawDataManager::TransformMultiDataToAlignment(QStringList PCList)
{
	#pragma region 先確認資料是正確的
	bool IsCurrent = true;

	// 確認是否有按照順序
	for (int i = 0; i < PCList.size(); i++)
		IsCurrent = IsCurrent & PCList[i].endsWith(QString::number(i + 1) + ".xyz");

	if (!IsCurrent)
	{
		cout << "資料可能不正確!" << endl;
		return;
	}
	#pragma endregion
	#pragma region 點雲拼接
	PointCloudArray.clear();
	//for (int i = 0; i < PCList.size(); i++)
	for (int i = 0; i < 5; i++)
	{
		PointCloudInfo pcInfo;
		pcInfo.ReadFromXYZ(PCList[i]);
		PointCloudArray.append(pcInfo);

		for (int j = 0; j < 3; j++)
			AlignmentPointCloud();
	}

	// SelectIndex
	SelectIndex = PointCloudArray.size() - 1;

	// 需更新
	PCWidgetUpdate();
	#pragma endregion
}
void RawDataManager::TransformMultiDataToPointCloud(QStringList rawDataList)
{
	#pragma region 先確認資料是正確的
	bool IsCurrent = true;
	IsCurrent = IsCurrent & rawDataList[0].endsWith("_C");

	// 確認是否有按照順序
	for (int i = 1; i < 13; i++)
		IsCurrent = IsCurrent & rawDataList[i].endsWith("_" + QString::number(i));

	if (!IsCurrent)
	{
		cout << "資料可能不正確!" << endl;
		return;
	}
	#pragma endregion
	#pragma region 接這按照順序轉點雲
	// 清空點雲
	PointCloudArray.clear();
	 
	// RawDataManager 轉換
	for (int i = 0; i < 13; i++)
	{
		// 轉換函式
		ReadRawDataFromFileV2(rawDataList[i]);
		TransformToIMG(false);
		TransformToOtherSideView();

		int index = rawDataList[i].lastIndexOf("/");
		QString SaveFileName = rawDataList[i].mid(0, index) + "/";
		if (i == 0)
			SaveFileName += "C";
		else
			SaveFileName += QString::number(i);
		SaveFileName += ".png";
		cv::imwrite(SaveFileName.toLocal8Bit().toStdString(), cudaV2.TransformToOtherSideView());

		// Quat
		QQuaternion quat;
		SavePointCloud(quat);

		/*if (i > 0)
			PointCloudArray[i].RotateConstantAngle(i - 1);*/
	}

	// 要存檔的
	for (int i = 0; i < 13; i++) 
	{
		int index = rawDataList[i].lastIndexOf("/");
		QString SaveFileName = rawDataList[i].mid(0, index) + "/";
		cout << SaveFileName.toLocal8Bit().toStdString() << endl;
		if (i == 0)
			SaveFileName += "C";
		else
			SaveFileName += QString::number(i);
		PointCloudArray[i].SaveXYZ(SaveFileName + ".xyz");
	}
	#pragma endregion
}
void RawDataManager::AverageErrorPC()
{
	#pragma region 算出中心點
	int TotalPointSize = 0;
	QVector3D tempP;
	for (int i = 0; i < PointCloudArray.size(); i++) 
	{
		int pointSize = PointCloudArray[i].Points.size();
		tempP += PointCloudArray[i].CenterPoints * pointSize;
		TotalPointSize += pointSize;

	}
	CenterPoint /= TotalPointSize;
	cout << CenterPoint.x() << " " << CenterPoint.y() << " " << CenterPoint.z() << endl;
	#pragma endregion
	#pragma region Debug 出結果

	#pragma endregion


	QVector3D GuessVec(0.0, 0.0, 0.0);

	QVector<QVector3D> ForTestPoint;
	for (int i = 0; i < PointCloudArray.size(); i++) {
		ForTestPoint.push_back(PointCloudArray[i].CenterPoints);
		cout << i << " : " << PointCloudArray[i].CenterPoints.x() << " " << PointCloudArray[i].CenterPoints.y() << " " << PointCloudArray[i].CenterPoints.z() << endl;
	}
	float* MatrixA = new float[PointCloudArray.size() * 3];
	float* MatrixB = new float[PointCloudArray.size()];
	float* params;

	for (int i = 0; i < PointCloudArray.size(); i++)
	{
		MatrixA[i * 3] = ForTestPoint[i].x();
		MatrixA[i * 3 + 1] = ForTestPoint[i].y();
		MatrixA[i * 3 + 2] = 1;
		MatrixB[i] = ForTestPoint[i].z();
	}

	Eigen::MatrixXf EigenMatrixA = Eigen::Map<Eigen::MatrixXf>(MatrixA, PointCloudArray.size(), 3);
	Eigen::MatrixXf EigenMatrixB = Eigen::Map<Eigen::MatrixXf>(MatrixB, PointCloudArray.size(), 1);

	EigenMatrixB = EigenMatrixA.transpose() * EigenMatrixB;
	EigenMatrixA = EigenMatrixA.transpose() * EigenMatrixA;

	Eigen::MatrixXf X = EigenMatrixA.householderQr().solve(EigenMatrixB);
	params = X.data();

	cout << params[0] << " " << params[1] << " " << params[2] << endl;

	PlaneZValue.setX(-params[0] * 5.0 - params[1] * 5.0 - params[2]);
	PlaneZValue.setY(-params[0] * 5.0 - params[1] * -5.0 - params[2]);
	PlaneZValue.setZ(-params[0] * -5.0 - params[1] * -5.0 - params[2]);
	PlaneZValue.setW(-params[0] * -5.0 - params[1] * 5.0 - params[2]);

	PlanePoint.push_back(QVector3D(5.0 + CenterPoint.x(), 5.0 + CenterPoint.y(), PlaneZValue.x() + CenterPoint.z()));
	PlanePoint.push_back(QVector3D(5.0 + CenterPoint.x(), -5.0 + CenterPoint.y(), PlaneZValue.y() + CenterPoint.z()));
	PlanePoint.push_back(QVector3D(-5.0 + CenterPoint.x(), -5.0 + CenterPoint.y(), PlaneZValue.z() + CenterPoint.z()));
	PlanePoint.push_back(QVector3D(-5.0 + CenterPoint.x(), 5.0 + CenterPoint.y(), PlaneZValue.w() + CenterPoint.z()));

}

// 網路
Mat RawDataManager::GetBoundingBox(Mat img, QVector2D& TopLeft, QVector2D& ButtomRight)
{
	#pragma region 抓出資料，並做模糊
	Mat imgGray;
	cvtColor(img, imgGray, COLOR_BGR2GRAY);
	blur(imgGray, imgGray, BlurSize);
	#pragma endregion
	#pragma region 根據閘值，去抓邊界
	// 先根據閘值，並抓取邊界
	Mat threshold_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	threshold(imgGray, threshold_output, BoundingThreshold, 255, THRESH_BINARY);
	findContours(threshold_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

	// 先給占存 Array
	vector<BoundingBoxDataStruct> dataInfo(contours.size());
	#pragma endregion
	#pragma region 抓出最大的框框
	// 抓出擬合的結果
	for (size_t img = 0; img < contours.size(); img++)
	{
		BoundingBoxDataStruct data;
		approxPolyDP(Mat(contours[img]), data.contoursPoly, 3, true);
		data.boundingRect = boundingRect(Mat(data.contoursPoly));

		// 加進陣列
		dataInfo[img] = data;
	}
	Mat drawing = img.clone();
	std::sort(dataInfo.begin(), dataInfo.end(), SortByContourPointSize);

	// 抓出最亮，且最大的
	int i = 0;
	vector<vector<Point>> contoursPoly(1);
	contoursPoly[0] = dataInfo[i].contoursPoly;

	// 確定這邊可以抓到
	drawContours(drawing, contoursPoly, (int)i, Scalar(255, 255, 255), 1, 8, vector<Vec4i>(), 0, Point());

	// 邊界
	Point tl = dataInfo[i].boundingRect.tl();
	tl.x = max(0, tl.x - BoundingOffset);
	tl.y = max(0, tl.y - BoundingOffset);
	Point br = dataInfo[i].boundingRect.br();
	br.x = min(DManager.prop.SizeZ, br.x + BoundingOffset);
	br.y = min(DManager.prop.SizeX, br.y + BoundingOffset);

	rectangle(drawing, tl, br, Scalar(0, 255, 255), 2, 8, 0);

	TopLeft.setX(tl.x);
	TopLeft.setY(tl.y);
	ButtomRight.setX(br.x);
	ButtomRight.setY(br.y);
	#pragma endregion
	return drawing;
}
bool RawDataManager::SortByContourPointSize(BoundingBoxDataStruct& c1, BoundingBoxDataStruct& c2)
{
	return c1.boundingRect.area() > c2.boundingRect.area();
}

// Helper Function
int RawDataManager::LerpFunction(int lastIndex, int lastValue, int nextIndex, int nextValue, int index)
{
	return (index - lastIndex) * (nextValue - lastValue) / (nextIndex - lastIndex) + lastValue;
}
QImage RawDataManager::Mat2QImage(cv::Mat const& src, int Type)
{
	cv::Mat temp;												// make the same cv::Mat
	if (Type == CV_8UC3)
		cvtColor(src, temp, CV_BGR2RGB);						// cvtColor Makes a copt, that what i need
	else if (Type == CV_32F)
	{
		src.convertTo(temp, CV_8U, 255);
		cvtColor(temp, temp, CV_GRAY2RGB);						// cvtColor Makes a copt, that what i need
	}
	QImage dest((const uchar *)temp.data, temp.cols, temp.rows, temp.step, QImage::Format_RGB888);
	dest.bits();												// enforce deep copy, see documentation 
																// of QImage::QImage ( const uchar * data, int width, int height, Format format )
	return dest;
}
string RawDataManager::MarshalString(System::String^ s)
{
	using namespace System::Runtime::InteropServices;
	const char* chars =
		(const char*)(Marshal::StringToHGlobalAnsi(s)).ToPointer();
	string os = chars;
	Marshal::FreeHGlobal(System::IntPtr((void*)chars));
	return os;
}
void RawDataManager::OCT_DataType_Transfrom(unsigned short *input, int inputlen, char *output)
{
	int outputlen_tmp = 0;
	for (int i = 0; i < inputlen; i++)
	{
		unsigned short tmp_short = input[i];
		tmp_short = tmp_short >> 8;

		output[outputlen_tmp] = input[i];
		outputlen_tmp++;

		output[outputlen_tmp] = tmp_short;
		outputlen_tmp++;
	}
}
void RawDataManager::ConvertQVector2Point3D(QVector<QVector3D>& q3DList, vector<Point3D>& pc)
{
	pc.clear();
	for (int i = 0; i < q3DList.size(); i++)
	{
		Point3D point3D;
		QVector3D currentP = q3DList[i];
		point3D.x() = currentP.x();
		point3D.y() = currentP.y();
		point3D.z() = currentP.z();
		pc.push_back(point3D);
	}
}
void RawDataManager::ConvertPoint3D2QVector(vector<Point3D>& pc, QVector<QVector3D>& q3DList)
{
	q3DList.clear();
	for (int i = 0 ; i< pc.size(); i++)
	{
		QVector3D p(
			pc[i].x(),
			pc[i].y(),
			pc[i].z()
		);
		q3DList.push_back(p);
	}
}
QMatrix4x4 RawDataManager::super4PCS_Align(vector<Point3D> *PC1, vector<Point3D> *PC2, float& FinalScore)
{
	clock_t t1, t2;
	t1 = clock();

	// Delta (see the paper).
	double delta = 0.1;

	// Estimated overlap (see the paper).
	//double overlap = 0.40;
	double overlap = 0.30;

	// Threshold of the computed overlap for termination. 1.0 means don't terminate
	// before the end.
	double thr = 0.35;

	// Maximum norm of RGB values between corresponded points. 1e9 means don't use.
	double max_color = 150;

	// Number of sampled points in both files. The 4PCS allows a very aggressive
	// sampling.
	int n_points = 500;

	// Maximum angle (degrees) between corresponded normals.
	double norm_diff = 20;
	
	bool use_super4pcs = true;

	// maximum per-dimension angle, check return value to detect invalid cases
	double max_angle = 20;

	int max_time_seconds = 1;
	//==========//

	//vector<Point3D> set1, set2;
	vector<Point2f> tex_coords1, tex_coords2;
	vector<Point3f> normals1, normals2;
	vector<tripple> tris1, tris2;
	vector<string> mtls1, mtls2;

	IOManager iomananger;


	// super4pcs matcher
	Match4PCSOptions options;


	Match4PCSBase::MatrixType *mat;
	mat = new Match4PCSBase::MatrixType;


	//bool overlapOk = options.configureOverlap(overlap, thr);
	bool overlapOk = options.configureOverlap(overlap, thr);
	/*if (!overlapOk) {
		cerr << "Invalid overlap configuration. ABORT" << endl;
		/// TODO Add proper error codes

	}*/

	options.sample_size = n_points;
	options.max_normal_difference = norm_diff;
	options.max_color_distance = max_color;
	options.max_time_seconds = max_time_seconds;
	options.delta = delta;
	options.max_angle = max_angle;

	Match4PCSOptions::Scalar Estimation;

	Estimation = options.getOverlapEstimation();
	cout << "getOverlapEstimation:" << Estimation << endl;

	//// Match and return the score (estimated overlap or the LCP).  
	typename Point3D::Scalar score = 0;
	int v;

	constexpr Utils::LogLevel loglvl = Utils::Verbose;

	//template <typename Visitor>
	using TrVisitorType = typename conditional <loglvl == Utils::NoLog,
		Match4PCSBase::DummyTransformVisitor,
		TransformVisitor>::type;

	Utils::Logger logger(loglvl);

	QMatrix4x4 matrix;
	matrix.setToIdentity();

	try {
		// 4PCS or Super4PCS
		if (use_super4pcs) {
			MatchSuper4PCS *matcher;
			matcher = new MatchSuper4PCS(options, logger);

			//cout << "Use Super4PCS" << endl;
			score = matcher->ComputeTransformation(*PC1, PC2, *mat);
		}
		else {
			Match4PCS *matcher;
			matcher = new Match4PCS(options, logger);
			//cout << "Use old 4PCS" << endl;
			score = matcher->ComputeTransformation(*PC1, PC2, *mat);
		}

		// 矩陣轉換
		matrix = QMatrix4x4((*mat).data());
	}
	catch (const std::exception& e) {
		cout << "[Error]: " << e.what() << '\n';
		cout << "Aborting with code -2 ..." << endl;
		return matrix;
	}
	catch (...) {
		std::cout << "[Unknown Error]: Aborting with code -3 ..." << std::endl;
		return matrix;
	}
	t2 = clock();
	//cout << "Score: " << score << endl;
	FinalScore = score;
	return matrix;
}
int RawDataManager::clamp(int value, int min, int max)
{
	return std::max(min, std::min(value, max));
}
