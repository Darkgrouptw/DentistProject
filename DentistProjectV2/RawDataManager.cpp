#include <QOpenGLWidget>				// 因為會跟 OpenCV 3 重圖
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
	GetQuaternion_Pointer		= bind(&RawDataManager::GetQuaternion,					this);
	CopySingleBorder_Pointer	= bind(&RawDataManager::CopySingleBorder,				this, placeholders::_1);
	ShakeDetect_Single_Pointer	= bind(&RawDataManager::ShakeDetect_Single,				this, placeholders::_1, placeholders::_2);
	ShakeDetect_Multi_Pointer	= bind(&RawDataManager::ShakeDetect_Multi,				this, placeholders::_1);
	SavePointCloud_Pointer		= bind(&RawDataManager::SavePointCloud,					this, placeholders::_1);
	AlignmentPointCloud_Pointer = bind(&RawDataManager::AlignmentPointCloud,			this);
	ShowImageIndex_Pointer		= bind(&RawDataManager::ShowImageIndex,					this, 60);

	// 傳進 Scan Thread 中
	Worker = gcnew ScanningWorkerThread(DManager.prop.SizeX);
	Worker->InitScanFunctionPointer(&ScanSingle_Pointer, &ScanMulti_Pointer, &TransforImage_Pointer, &GetQuaternion_Pointer);
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
}

// UI 相關
void RawDataManager::SendUIPointer(QVector<QObject*> UIPointer)
{
	// 確認是不是有多傳，忘了改的
	assert(UIPointer.size() == 19 && "UI 對應有問題!!");
	ImageResult				= (QLabel*)UIPointer[0];
	BorderDetectionResult	= (QLabel*)UIPointer[1];
	NetworkResult			= (QLabel*)UIPointer[2];
	OtherSideResult			= (QLabel*)UIPointer[18];

	// 後面兩個是 給 ScanThread
	QSlider* slider			= (QSlider*)UIPointer[3];
	QPushButton* scanButton = (QPushButton*)UIPointer[4];
	QLineEdit* saveLineEdt	= (QLineEdit*)UIPointer[5];
	
	// OpenGL
	DisplayPanel			= UIPointer[6];

	// 角度顯示
	// 8  ~ 11	=>	Quaternion
	// 12 ~ 14	=>	QSlider
	// 15 ~ 17	=>	QLabel
	PCIndex					= (QComboBox*)UIPointer[7];
	for (int i = 0; i < 4; i++)
	{
		QLineEdit* tempQuaternion = (QLineEdit*)UIPointer[8 + i];
		QuaternionLinEditArray.push_back(tempQuaternion);
	}
	for (int i = 0; i < 3;i++)
	{
		QSlider* tempSlider = (QSlider*)UIPointer[12 + i];
		QLabel* tempLabel	= (QLabel*)UIPointer[15 + i];
		EulerBarArray.push_back(tempSlider);
		EulerTextArray.push_back(tempLabel);
	}

	Worker->InitUIPointer(slider, scanButton, saveLineEdt);
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
	#pragma region 材扁圖
	Mat result = cudaV2.TransformToOtherSideView();
	QImage qreulst = Mat2QImage(result, CV_8UC3);
	OtherSideResult->setPixmap(QPixmap::fromImage(qreulst));
	#pragma endregion
	#pragma region 結束時間
	#ifdef SHOW_TRCUDAV2_TRANSFORM_TIME
	endT = clock();
	cout << "材扁圖轉換時間: " << (endT - startT) / (double)(CLOCKS_PER_SEC) << "s" << endl;
	#endif
	#pragma endregion
}
QQuaternion RawDataManager::GetQuaternion()
{
	return bleManager.GetQuaternionFromDevice();
}
void RawDataManager::SetScanOCTMode(bool IsStart, QString* EndText, bool NeedSave_Single_RawData, bool NeedSave_Multi_RawData, bool NeedSave_ImageData, bool AutoDelete_ShakeData)
{
	Worker->SetParams(EndText, NeedSave_Single_RawData, NeedSave_Multi_RawData, NeedSave_ImageData, AutoDelete_ShakeData);
	Worker->SetScanModel(IsStart);
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
bool RawDataManager::ShakeDetect_Multi(bool ShowDebugMessage)
{
	return cudaV2.ShakeDetect_Multi(ShowDebugMessage);
}
void RawDataManager::SavePointCloud(QQuaternion quat)
{
	#pragma region 創建 Array
	// 更新
	IsLockPC = true;

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
	//rotationMatrix.rotate(90, 1, 0, 0);
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

	// DisplayPanel
	((OpenGLWidget*)DisplayPanel)->UpdatePC();
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
	PCWidgetUpdate();

	// 結束
	IsLockPC = false;
	#pragma endregion
}
void RawDataManager::AlignmentPointCloud()
{
	// 點雲拼接
	if (PointCloudArray.size() > 1)
	{
		// 拿最後一篇跟其他拼接
		int LastID = PointCloudArray.size() - 1;
		vector<Point3D> NewPC, LastPC;
		ConvertQVector2Point3D(PointCloudArray[LastID].Points, NewPC);
		ConvertQVector2Point3D(PointCloudArray[LastID - 1].Points, LastPC);

		// 轉換 Matrix
		float score = 0;
		super4PCS_Align(&LastPC, &NewPC, score);
		ConvertPoint3D2QVector(NewPC, PointCloudArray[LastID].Points);

		IsLockPC = true;

		// 這邊再去做判斷
		// 如果分數小於一個 Threshold 那就丟掉
		if (score < AlignScoreThrshold)
		{
			PointCloudArray.removeLast();
			SelectIndex--;
			PCWidgetUpdate();
		}

		((OpenGLWidget*)DisplayPanel)->UpdatePC();
		IsLockPC = false;
	}
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

			cout << "Use Super4PCS" << endl;
			score = matcher->ComputeTransformation(*PC1, PC2, *mat);
		}
		else {
			Match4PCS *matcher;
			matcher = new Match4PCS(options, logger);
			cout << "Use old 4PCS" << endl;
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
	cout << "Score: " << score << endl;
	FinalScore = score;
	return matrix;
}
void RawDataManager::PCWidgetUpdate()
{
	PCIndex->clear();
	for (int i = 0; i < PointCloudArray.size(); i++)
	{
		QString tempText = QString::number(i) + " (" + QString::number(PointCloudArray[i].Points.size()) + ")";
		PCIndex->addItem(tempText);
	}

	// 重新設定 Rotation
	QQuaternion quat = QuaternionList[SelectIndex];
	QuaternionLinEditArray[0]->setText(QString::number(quat.x()));
	QuaternionLinEditArray[1]->setText(QString::number(quat.y()));
	QuaternionLinEditArray[2]->setText(QString::number(quat.z()));
	QuaternionLinEditArray[3]->setText(QString::number(quat.scalar()));

	QVector3D angle = quat.toEulerAngles();
	int AngleX = (int)(angle[0] + 360) % 360;
	int AngleY = (int)(angle[1] + 360) % 360;
	int AngleZ = (int)(angle[2] + 360) % 360;
	EulerBarArray[0]->setValue(AngleX);
	EulerBarArray[1]->setValue(AngleY);
	EulerBarArray[2]->setValue(AngleZ);

	EulerTextArray[0]->setText(QString::number(AngleX));
	EulerTextArray[1]->setText(QString::number(AngleY));
	EulerTextArray[2]->setText(QString::number(AngleZ));
}