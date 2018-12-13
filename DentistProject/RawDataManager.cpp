#include "RawDataManager.h"

RawDataManager::RawDataManager()
{
	cout << "OpenCV Version: " << CV_VERSION << endl;
	DManager.ReadCalibrationData();
	cudaBorder.Init(250, 1024);
}
RawDataManager::~RawDataManager()
{
}

// UI 相關
void RawDataManager::SendUIPointer(QVector<QObject*> UIPointer)
{
	// 確認是不是有多傳，忘了改的
	assert(UIPointer.size() == 3);
	ImageResult			= (QLabel*)UIPointer[0];
	NetworkResult		= (QLabel*)UIPointer[1];
	FinalResult			= (QLabel*)UIPointer[2];
}
void RawDataManager::ShowImageIndex(int index)
{
	if (60 <= index && index <= 200 && ImageResultArray.size() > 0)
	{
		QImage Pixmap_ImageResult = QImageResultArray[index - 60];
		ImageResult->setPixmap(QPixmap::fromImage(Pixmap_ImageResult));

		//QImage Pixmap_NetworkResult = Mat2QImage(FastBorderResultArray[index - 60], CV_8UC3);
		//NetworkResult->setPixmap(QPixmap::fromImage(Pixmap_NetworkResult));

		// 如果有東西的話才顯示 Network 預測的結果
		if (CombineResultArray.size() > 0)
		{
			QImage Pixmap_FinalResult = QCombineResultArray[index - 60];
			FinalResult->setPixmap(QPixmap::fromImage(Pixmap_FinalResult));
		}
	}
}

// OCT 相關的步驟
void RawDataManager::ReadRawDataFromFile(QString FileName)
{
	clock_t startT, endT;
	startT = clock();

	QFile inputFile(FileName);
	if (!inputFile.open(QIODevice::ReadOnly))
	{
		cout << "Raw Data 讀取錯誤" << endl;
		return;
	}
	else
		cout << "讀取 Raw Data: " << FileName.toLocal8Bit().constData() << endl;

	int bufferSize = inputFile.size() / sizeof(quint8);
	//cout << "Size : " << bufferSize << endl;

	QDataStream qData(&inputFile);
	buffer.clear();
	buffer.resize(bufferSize);
	qData.readRawData(buffer.data(), bufferSize);


	// throw useless data
	int scanline = (DManager.rawDP.size_Z * 2) * DManager.rawDP.size_Y * 2;
	for (int i = scanline; i < buffer.size(); i += scanline)
		buffer.remove(i, scanline);
	DManager.rawDP.size_X = 2 * buffer.size() / ((DManager.rawDP.size_Z * 2) * DManager.rawDP.size_Y * 2);

	inputFile.close();
	endT = clock();

	cout << "ReadRawData done t: " << (endT - startT) / (double)(CLOCKS_PER_SEC) << " sec" << endl;

	RawDataProperty *tmpRDP = &DManager.rawDP;
	theTRcuda.RawToPointCloud(buffer.data(), buffer.size(), tmpRDP->size_Y, tmpRDP->size_Z, 1);
}
void RawDataManager::ScanDataFromDevice(QString SaveFileName, bool NeedSave_RawData)
{
	#pragma region 初始化裝置
	OCT64::OCT64::Init(
		4,
		OCT_DeviceID
	);
	#pragma endregion
	#pragma region 開 Port
	SerialPort port(gcnew System::String(OCTDevicePort.c_str()), 9600);
	port.Open();

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
	System::String^ SaveFileName_C = gcnew System::String(SaveFileName.toStdString().c_str());

	OCT64::OCT64::StartCap(
		OCT_DeviceID,						// 裝置 ID
		OCT_HandleOut,						// Handle (要傳給 Scan 的)
		LV_65,								// ?
		SampleCount,						// 2048
		OCT_DataLen,						// 資料長度
		NeedSave_RawData,							// 這個好像是要步要 output
		SaveFileName_C,						// 儲存位置
		OCT_ErrorBoolean,					// 是否要有 Error
		ErrorString							// 錯誤訊息
	);
	port.RtsEnable = true;
	// cout << "OCT StartCap Error String: " << MarshalString(ErrorString) << endl;
	// StartCap(deviceID, tmp_Handle, LV_65, SampRec, tmp_ByteLen, Savedata, SaveName, tmp_ErrorBoolean, ErrorString, ErrorString_len_in, tmp_ErrorString_len_out);

	// 要接的 Array
	cli::array<unsigned short>^ OutDataArray = gcnew cli::array<unsigned short>(OCT_PIC_SIZE);				// 暫存的 Array
	unsigned short* Final_OCT_Array = new unsigned short[OCT_PIC_SIZE * 125];								// 取值得 Array
	unsigned short* Temp_OCT_Pointer = Final_OCT_Array;														// 暫存，因為上面那個會一直位移 (別問我，前人就這樣寫= =)
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
		// cout << "Scan Error String: " << MarshalString(ErrorString) << endl;
		//ScanADC(HandleOut, AllDatabyte, ArrSize, ByteLen, outarr, OutArrLenIn, tmp_OutArrLenOut, ErrorString, ErrorString_len_in, tmp_ErrorString_len_out);

		// cli Array 轉到 manage array
		pin_ptr<unsigned short> pinPtrArray = &OutDataArray[OutDataArray->GetLowerBound(0)];
		memcpy(Final_OCT_Array, pinPtrArray, sizeof(unsigned short) * OCT_PIC_SIZE);
		Final_OCT_Array += OCT_PIC_SIZE;																	// Offset 一段距離
		//memcpy(final_oct_arr, outarr, sizeof(unsigned short) * PIC_SIZE);
		//final_oct_arr = final_oct_arr + PIC_SIZE;

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
	OCT_DataType_Transfrom(Temp_OCT_Pointer, OCT_PIC_SIZE * 125, Final_OCT_Char);
	// unsigned_short_to_char(interator, PIC_SIZE * 125, final_oct_char);

	// 要將資料轉到 DManager.rawDP 上
	vector<char> VectorScan(Final_OCT_Char, Final_OCT_Char + OCT_PIC_SIZE * 250);
	theTRcuda.RawToPointCloud(VectorScan.data(), VectorScan.size(), 250, 2048);
	#pragma endregion
	#pragma region 刪除 New 出來的 Array
	delete Temp_OCT_Pointer;
	delete Final_OCT_Char;
	#pragma endregion
	/*pin_ptr<int32_t> tmp_deviceID = &deviceID;
	InitADC(4, tmp_deviceID);
	clock_t scan_t1 = clock();

	float LV_65 = 65;
	char SaveName[] = "V:\\OCT20170928";
	unsigned int SampRec = 2048;

	char SaveName2[1024];
	std::string name_title = "V:\\";

	(*out_fileName) = name_title + (*out_fileName);

	strcpy(SaveName2, out_fileName->c_str());
	SerialPort port("COM6", 9600);
	port.Open();

	if (!port.IsOpen) {
		std::cout << "COM6 fail to open!" << std::endl;
	}
	Sleep(100);

	Savedata = true;
	ErrorBoolean = false;
	ByteLen = 1;


	pin_ptr<int32_t> tmp_ErrorString_len_out = &ErrorString_len_out;
	pin_ptr<uint32_t> tmp_Handle = &HandleOut;
	pin_ptr<uint32_t> tmp_ByteLen = &ByteLen;
	pin_ptr<LVBoolean> tmp_ErrorBoolean = &ErrorBoolean;


	StartCap(deviceID, tmp_Handle, LV_65, SampRec, tmp_ByteLen, Savedata, SaveName, tmp_ErrorBoolean, ErrorString, ErrorString_len_in, tmp_ErrorString_len_out);
	port.RtsEnable = true;
	/////////////////////////////////////////////////////////H_StartCap///////////////////////////////////////////////


	/////////////////////////////////////////////////////////H_ScanADC///////////////////////////////////////////////
	AllDatabyte = ByteLen * 2;
	ArrSize = new unsigned short[PIC_SIZE];
	outarr = new unsigned short[PIC_SIZE];
	final_oct_arr = new unsigned short[PIC_SIZE * 125];
	final_oct_char = new char[PIC_SIZE * 250];

	OutArrLenIn = 2048 * 500 * 2;
	OutArrLenOut = 0;
	pin_ptr<int32_t> tmp_OutArrLenOut = &OutArrLenOut;
	int pic_count = 0;
	clock_t t3, t4, t5;
	bool change_point_cloud = false;

	//Sleep(100);

	t3 = clock();

	unsigned short *interator;
	interator = new unsigned short;
	interator = final_oct_arr;
	//final_oct_arr = final_oct_arr + PIC_SIZE * 62;


	while (pic_count < 125) {
		ScanADC(HandleOut, AllDatabyte, ArrSize, ByteLen, outarr, OutArrLenIn, tmp_OutArrLenOut, ErrorString, ErrorString_len_in, tmp_ErrorString_len_out);
		memcpy(final_oct_arr, outarr, sizeof(unsigned short) * PIC_SIZE);

		pic_count++;

		final_oct_arr = final_oct_arr + PIC_SIZE;

	}
	AboutADC(deviceID);
	port.RtsEnable = false;
	port.Close();

	t4 = clock();

	unsigned_short_to_char(interator, PIC_SIZE * 125, final_oct_char);
	t5 = clock();

	std::vector<char> tmp_char(final_oct_char, final_oct_char + PIC_SIZE * 250);

	std::cout << "to array time:" << (t4 - t3) / (double)(CLOCKS_PER_SEC) << "s" << std::endl;
	std::cout << "short to char time:" << (t5 - t4) / (double)(CLOCKS_PER_SEC) << "s" << std::endl;

	theTRcuda->RawToPointCloud(tmp_char.data(), tmp_char.size(), 250, 2048);
	scan_count++;

	int PCidx, Mapidx;
	float ratio = 1;
	float zRatio = DManager->zRatio;
	PointCloudArray tmpPC_T;

	for (int x = 2; x < theTRcuda->VolumeSize_X - 2; x++)
	{
		for (int y = 0; y < theTRcuda->VolumeSize_Y; y++)
		{
			if (x > DManager->Mapping_X || y > DManager->Mapping_Y)
				continue;

			Mapidx = ((y * theTRcuda->sample_Y * theTRcuda->VolumeSize_X) + x) * theTRcuda->sample_X;
			PCidx = ((x * theTRcuda->VolumeSize_Y) + y) * theTRcuda->VolumeSize_Z;
			if (theTRcuda->PointType[PCidx + 3] == 1 && theTRcuda->PointType[PCidx + 1] != 0)
			{
				GlobalRegistration::Point3D tmp;
				PointData tmp_Data;

				tmp_Data.Position.x() = DManager->MappingMatrix[Mapidx * 2 + 1] * ratio + 0.2;
				tmp_Data.Position.y() = DManager->MappingMatrix[Mapidx * 2] * ratio;
				tmp_Data.Position.z() = theTRcuda->PointType[PCidx + 1] * zRatio / theTRcuda->VolumeSize_Z * ratio;

				tmpPC_T.mPC.push_back(tmp_Data);
				if (x == 124) {
					point_z_124 = point_z_124 + tmp_Data.Position.z();
				}
				if (x == 125) {
					point_z_125 = point_z_125 + tmp_Data.Position.z();
				}
			}
		}
	}

	if (tmpPC_T.mPC.size() > 10000) {
		(*PointCloudArr).push_back(tmpPC_T);
		PointCloud_idx_show = (*PointCloudArr).size() - 1;

		(*PointCloudArr)[PointCloud_idx_show].mPC_noGyro = tmpPC_T.mPC;
		std::cout << "tmpPC_T.mPC.size(): " << tmpPC_T.mPC.size() << std::endl;

		std::cout << "point_z_124 Total: " << point_z_124 << std::endl;
		std::cout << "point_z_125 Total: " << point_z_125 << std::endl;

		Find_max_min();
		draw_before_mapping();

		std::cout << "now point cloud size:" << (*PointCloudArr).size() << std::endl;
		std::cout << "Scan count:" << scan_count << std::endl;
	}

	is_camera_move = true;

	if ((*PointCloudArr).size() == 1) {
		first_t = clock();
	}
	else if ((*PointCloudArr).size() == 2) {
		Rotate_cloud();
		std::cout << "now idx:" << PointCloud_idx_show << "     now size is:" << (*PointCloudArr).size() << "   mPC size is:" << (*PointCloudArr)[PointCloud_idx_show].mPC.size() << "   degree is:" << (*PointCloudArr)[PointCloud_idx_show].cloud_degree_arr;
	}
	else if ((*PointCloudArr).size()>2) {
		Rotate_cloud();
		Find_min_quat3();
		std::cout << "now idx:" << PointCloud_idx_show << "     now size is:" << (*PointCloudArr).size() << "   mPC size is:" << (*PointCloudArr)[PointCloud_idx_show].mPC.size() << "   degree is:" << (*PointCloudArr)[PointCloud_idx_show].cloud_degree_arr;
	}
	clear_PC_buffer();
	color_img();

	delete[] interator;
	delete[] final_oct_char;
	delete[] ArrSize;
	delete[] outarr;

	clock_t scan_t2 = clock();
	full_scan_time = (scan_t2 - scan_t1) / (double)(CLOCKS_PER_SEC);
	all_time = all_time + full_scan_time;

	std::cout << "full_scan_time:" << full_scan_time << "s" << std::endl;
	std::cout << "all_time:" << all_time << "s" << std::endl;*/
}
void RawDataManager::TranformToIMG(bool NeedSave_Image = false)
{
	#pragma region 開始時間
	clock_t startT, endT;
	startT = clock();
	#pragma endregion
	#pragma region 清空其他 Array
	// 如果跑出結果是全黑的，那有可能是顯卡記憶體不夠的問題
	ImageResultArray.clear();
	SmoothResultArray.clear();
	CombineResultArray.clear();

	QImageResultArray.clear();
	QSmoothResultArray.clear();
	QCombineResultArray.clear();

	// Point Cloud 相關
	PointCloudArray.clear();
	#pragma endregion
	#pragma region Init 二維陣列 (將資料傳到 CudaBorder 裡面)
	float *_OCTData = new float[theTRcuda.VolumeSize_Y * theTRcuda.VolumeSize_Z];
	float** OCTData = new float*[theTRcuda.VolumeSize_Y];
	memset(_OCTData, 0, sizeof(float) * theTRcuda.VolumeSize_Y * theTRcuda.VolumeSize_Z);

	float *_OCTDataAvg = new float[theTRcuda.VolumeSize_Y * theTRcuda.VolumeSize_Z];
	float** OCTDataAvg = new float*[theTRcuda.VolumeSize_Y];
	memset(_OCTDataAvg, 0, sizeof(float) * theTRcuda.VolumeSize_Y * theTRcuda.VolumeSize_Z);

	// 將資料放進去
	for (int i = 0; i < theTRcuda.VolumeSize_Y; i++)
	{
		OCTData[i]		= &_OCTData[i * theTRcuda.VolumeSize_Z];
		OCTDataAvg[i]	= &_OCTDataAvg[i * theTRcuda.VolumeSize_Z];
	}
	#pragma endregion
	#pragma region 抓取資訊塞到圖裡面
	// 暫存點雲
	QVector<QVector3D> CurrentPointCloud;

	// 取 60 ~ 200
	// for (int x = 0; x < theTRcuda.VolumeSize_X; x++)
	int TheCudaSize = 64000000;
	for (int x = 60; x <= 200; x++)
	{
		// Mat
		cv::Mat ImageResult;		// = cv::Mat(theTRcuda.VolumeSize_Y, theTRcuda.VolumeSize_Z, CV_32F, cv::Scalar(0, 0, 0));
		cv::Mat SmoothResult;		//= cv::Mat(theTRcuda.VolumeSize_Y, theTRcuda.VolumeSize_Z, CV_32F, cv::Scalar(0, 0, 0));
		cv::Mat CombineResult;		//= cv::Mat(theTRcuda.VolumeSize_Y, theTRcuda.VolumeSize_Z, CV_32F, cv::Scalar(0, 0, 0));

		// QImage
		QImage QImageResult;
		QImage QSmoothResult;
		QImage QCombineResult;

		// 先 Mapping 資料
		cudaBorder.MappingData(theTRcuda.VolumeData, TheCudaSize, OCTData, x);
		ImageResult = cudaBorder.SaveDataToImage(OCTData, false);
		QImageResult = Mat2QImage(ImageResult, CV_8UC3);

		cudaBorder.MappingData(theTRcuda.VolumeDataAvg, TheCudaSize, OCTDataAvg, x);
		SmoothResult = cudaBorder.SaveDataToImage(OCTDataAvg, false);
		QSmoothResult = Mat2QImage(SmoothResult, CV_8UC3);

		cudaBorder.GetBorderFromCuda(OCTDataAvg);
		CombineResult = cudaBorder.SaveDataToImage(OCTDataAvg, true);
		QCombineResult = Mat2QImage(CombineResult, CV_8UC3);

		if (NeedSave_Image)
		{
			// 原圖
			cv::imwrite("Images/OCTImages/origin_v2/" + std::to_string(x) + ".png", ImageResult);

			// Smooth 影像
			cv::imwrite("Images/OCTImages/smooth_v2/" + std::to_string(x) + ".png", SmoothResult);

			// Combine 結果圖
			cv::imwrite("Images/OCTImages/combine_v2/" + std::to_string(x) + ".png", CombineResult);
		}
		else

		// 暫存到陣列裡 (Mat)
		ImageResultArray.push_back(ImageResult);
		SmoothResultArray.push_back(SmoothResult);
		CombineResultArray.push_back(CombineResult);
		
		// 暫存到陣列裡 (QImage)
		QImageResultArray.push_back(QImageResult);
		QSmoothResultArray.push_back(QSmoothResult);
		QCombineResultArray.push_back(QCombineResult);
	}

	// 加入 Point Cloud 的陣列
	PointCloudArray.push_back(CurrentPointCloud);
	#pragma endregion
	#pragma region 刪除 Array
	delete _OCTData;
	delete OCTData;
	delete _OCTDataAvg;
	delete OCTDataAvg;
	#pragma endregion
	#pragma region 結束時間
	endT = clock();
	#pragma endregion
	#pragma region Final Output
	if (NeedSave_Image)
		cout << "有存出圖片";
	else
		cout << "無存出圖片";
	cout << "，轉圖檔完成: " << (endT - startT) / (double)(CLOCKS_PER_SEC) << "s" << endl;
	#pragma endregion
}
bool RawDataManager::ShakeDetect(QMainWindow *main, bool IsShowForDebug)
{
	clock_t startT, endT;
	startT = clock();

	// 錯誤判斷，判斷進入這個 Function 一定要有資料
	if (ImageResultArray.size() == 0)
	{
		QMessageBox::critical(main, codec->toUnicode("Function 錯誤"), codec->toUnicode("沒有資料可以執行!!"));
		return false;
	}
	cv::Mat FirstImage = SmoothResultArray[124 - 60];
	FirstImage.convertTo(FirstImage, CV_8U, 255);
	cv::Mat LastImage = SmoothResultArray[125 - 60];
	LastImage.convertTo(LastImage, CV_8U, 255);

	cv::Mat SmallFirstMat(cv::Size(500, 250), CV_8U);
	cv::Mat SmallLastMat(cv::Size(500, 250), CV_8U);

	cv::threshold(FirstImage(cv::Rect(0, 0, 500, 250)), SmallFirstMat, 20, 255, cv::THRESH_BINARY);
	cv::threshold(LastImage(cv::Rect(0, 0, 500, 250)), SmallLastMat, 20, 255, cv::THRESH_BINARY);

	cv::Size size(SmallFirstMat.cols / 4, SmallFirstMat.rows / 4);
	cv::resize(SmallFirstMat, SmallFirstMat, size);
	cv::resize(SmallLastMat, SmallLastMat, size);


	float ZerosCount1 = 1.0f - (float)cv::countNonZero(SmallFirstMat) / SmallFirstMat.rows / SmallFirstMat.cols;
	float ZerosCount2 = 1.0f - (float)cv::countNonZero(SmallLastMat) / SmallLastMat.rows / SmallLastMat.cols;

	double PSNR_Value = PSNR(SmallFirstMat, SmallLastMat);
	//double PSNR_BlackFirst = PSNR
	endT = clock();

	if (IsShowForDebug)
	{
		imshow("First", SmallFirstMat);
		imshow("Last", SmallLastMat);

		cout << "PSNR: " << PSNR_Value << endl;
		cout << "Zero1 Count: " << ZerosCount1 << endl;
		cout << "Zero2 Count: " << ZerosCount2 << endl;
		cout << "晃動判斷時間: " << (endT - startT) / (double)(CLOCKS_PER_SEC) << " sec" << endl;
	}

	// PSNR && 有效區域要大於某一個值
	if (PSNR_Value > OCT_PSNR_Threshold &&
		(ZerosCount1 < OCT_UsefulData_Threshold && ZerosCount2 < OCT_UsefulData_Threshold))
		return true;
	return false;
}
void RawDataManager::WriteRawDataToFile(QString DirLocation)
{
	for (int x = 0; x < theTRcuda.VolumeSize_X; x++)
	{
		// 開啟檔案
		QFile file(DirLocation + "/" + QString::number(x) + ".txt");
		assert(file.open(QIODevice::WriteOnly));

		QTextStream ts(&file);

		// Map 參數
		for (int row = 0; row < theTRcuda.VolumeSize_Y; row++)
		{
			// 這個 For 迴圈是算每一個結果
			// 也就是去算深度底下，1024 個結果
			for (int col = 1; col < theTRcuda.VolumeSize_Z; col++)
			{
				int tmpIdx = ((x * theTRcuda.VolumeSize_Y) + row) * theTRcuda.VolumeSize_Z + col;
				ts << theTRcuda.VolumeDataAvg[tmpIdx] << "\t";
				//ts << theTRcuda.VolumeData[tmpIdx] << "\t";
			}
			ts << "\n";
		}
		file.close();

		// 輸出進度
		if (x > 0)
			cout << "\r";
		cout << "輸出進度:" << (x + 1) << " / " << theTRcuda.VolumeSize_X;
	}
	cout << endl << "完成!!" << endl;
}

// Netowrk 相關的 Function
QVector<cv::Mat> RawDataManager::GenerateNetworkData()
{
	// 這邊要先確保他會大於 0
	assert(ImageResultArray.size() > 0);

	QVector<cv::Mat> InputData;
	int ColTimes = ImageResultArray[0].rows / NetworkCutRow;
	for (int i = 1; i <ImageResultArray.size() - 1; i++)
		for (int j = 0; j < ColTimes; j++)
		{
			// 宣告
			cv::Mat InputMat(cv::Size(NetworkCutCol, NetworkCutRow), CV_8UC3);
			vector<cv::Mat> channelsData;

			// 拆成 BGR Channel
			cv::split(InputMat, channelsData);

			// B G R 塞值進去
			channelsData[0] = ImageResultArray[i - 1](cv::Rect(0, j * NetworkCutRow, NetworkCutCol, NetworkCutRow));
			channelsData[1] = ImageResultArray[i + 0](cv::Rect(0, j * NetworkCutRow, NetworkCutCol, NetworkCutRow));
			channelsData[2] = ImageResultArray[i + 1](cv::Rect(0, j * NetworkCutRow, NetworkCutCol, NetworkCutRow));

			// 這邊是
			cv::merge(channelsData, InputMat);
			InputData.push_back(InputMat);
		}
	// Debug 圖片測試
	//cv::imshow("TestImage", InputData[0]);
	//cv::waitKey(0);
	return InputData;
}
void RawDataManager::SetPredictData(QVector<cv::Mat> PredictArray)
{
	int ColTimes = ImageResultArray[0].rows / NetworkCutRow;

	// 這邊要確保結果是剛剛好的 (也就是 5 張合成一張完整圖的話，要確保傳進來的結果大小是 5 的倍數)
	// 然後還要大於 0
	assert((PredictArray.size() % ColTimes == 0) && (PredictArray.size() > 0));

	CombineResultArray.clear();
	for (int i = 0; i < PredictArray.size(); i += ColTimes)
	{
		cv::Mat result = cv::Mat(cv::Size(ImageResultArray[0].cols, ImageResultArray[0].rows), CV_8UC3);
		for (int j = 0; j < ColTimes; j++)
			PredictArray[i + j].copyTo(result(cv::Rect(0, j * NetworkCutRow, NetworkCutCol, NetworkCutRow)));
		CombineResultArray.push_back(result);
	}
}

//////////////////////////////////////////////////////////////////////////
// Helper Function
//////////////////////////////////////////////////////////////////////////
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
