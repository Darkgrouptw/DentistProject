#include "RawDataManager.h"

RawDataManager::RawDataManager()
{
	cout << "OpenCV Version: " << CV_VERSION << endl;
	DManager.ReadCalibrationData();
}
RawDataManager::~RawDataManager()
{
}

// UI 相關
void RawDataManager::SendUIPointer(QVector<QObject*> UIPointer)
{
	// 確認是不是有多傳，忘了改的
	assert(UIPointer.size() == 2);
	ImageResult = (QLabel*)UIPointer[0];
	FinalResult = (QLabel*)UIPointer[1];
}
void RawDataManager::ShowImageIndex(int index)
{
	if (0 <= index && index < 250 && ImageResultArray.size() > 0)
	{
		QImage Pixmap_ImageResult = Mat2QImage(ImageResultArray[index], CV_32F);
		QImage Pixmap_FinalResult = Mat2QImage(CombineTestArray[index], CV_8UC3);
		ImageResult->setPixmap(QPixmap::fromImage(Pixmap_ImageResult));
		FinalResult->setPixmap(QPixmap::fromImage(Pixmap_FinalResult));
	}
}

// OCT 相關的步驟
void RawDataManager::ReadRawDataFromFile(QString FileName)
{
	clock_t t1, t2;
	t1 = clock();

	QFile inputFile(FileName);
	if (!inputFile.open(QIODevice::ReadOnly))
	{
		std::cout << "Raw Data 讀取錯誤" << std::endl;
		return;
	}

	int bufferSize = inputFile.size() / sizeof(quint8);
	std::cout << "Size : " << bufferSize << std::endl;

	QDataStream qData(&inputFile);
	buffer.clear();
	buffer.resize(bufferSize);
	qData.readRawData(buffer.data(), bufferSize);


	// throw useless data
	int scanline = (DManager.rawDP.size_Z * 2) * DManager.rawDP.size_Y * 2;
	for (int i = scanline; i < buffer.size(); i += scanline)
		buffer.remove(i, scanline);

	std::cout << "size of buffer : " << buffer.size() << std::endl;
	DManager.rawDP.size_X = 2 * buffer.size() / ((DManager.rawDP.size_Z * 2) * DManager.rawDP.size_Y * 2);

	inputFile.close();
	t2 = clock();

	std::cout << "ReadRawData done t: " << (t2 - t1) / (double)(CLOCKS_PER_SEC) << " s" << std::endl;

	RawDataProperty *tmpRDP = &DManager.rawDP;
	theTRcuda.RawToPointCloud(buffer.data(), buffer.size(), tmpRDP->size_Y, tmpRDP->size_Z, 1);
}
void RawDataManager::ScanDataFromDevice(QString SaveFileName, bool NeedSave)
{
	//string SaveName = "V:/OCT20170928";

	#pragma region 初始化裝置
	OCT64::OCT64::Init(
		4,
		OCT_DeviceID
	);
	#pragma endregion
	#pragma region 開 Port
	SerialPort port(gcnew System::String(OCTDevicePort.c_str()), 9600);
	port.Open();

	if (!port.IsOpen)
	{
		cout << "OCT 的 COM 打不開!!" << endl;
		return;
	}

	// 先休眠
	Thread::Sleep(100);
	port.RtsEnable = true;
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
		NeedSave,							// 這個好像是要步要 output
		SaveFileName_C,						// 儲存位置
		OCT_ErrorBoolean,					// 是否要有 Error
		ErrorString							// 錯誤訊息
	);
	// cout << "OCT StartCap Error String: " << MarshalString(ErrorString);
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
	OCT_DataType_Transfrom(Temp_OCT_Pointer, OCT_PIC_SIZE * 125, Final_OCT_Char);
	// unsigned_short_to_char(interator, PIC_SIZE * 125, final_oct_char);

	// 要將資料轉到 DManager.rawDP 上
	vector<char> VectorScan(Final_OCT_Char, Final_OCT_Char + OCT_PIC_SIZE * 250);
	theTRcuda.RawToPointCloud(VectorScan.data(), VectorScan.size(), 250, 2048);
	#pragma endregion
	#pragma region 刪除新創的 New Array
	delete Final_OCT_Array;
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
void RawDataManager::TranformToIMG(bool NeedSave = false)
{
	// 如果跑出結果是全黑的，那有可能是顯卡記憶體不夠的問題
	ImageResultArray.clear();
	SmoothResultArray.clear();
	// FastLabelArray.clear();
	CombineTestArray.clear();

	// bool DoFastLabel = true;

	//////////////////////////////////////////////////////////////////////////
	// 這邊底下是舊的 Code
	// 不是我寫的 QAQ
	//////////////////////////////////////////////////////////////////////////
	// 取 60 ~ 200
	for (int x = 0; x < theTRcuda.VolumeSize_X; x++) 
	//for (int x = 60; x <= 200; x++)
	{
		// Mat
		cv::Mat ImageResult = cv::Mat(theTRcuda.VolumeSize_Y, theTRcuda.VolumeSize_Z, CV_32F, cv::Scalar(0, 0, 0));
		cv::Mat SmoothResult = cv::Mat(theTRcuda.VolumeSize_Y, theTRcuda.VolumeSize_Z, CV_32F, cv::Scalar(0, 0, 0));
		//cv::Mat FastLabel = cv::Mat(theTRcuda.VolumeSize_Y, theTRcuda.VolumeSize_Z, CV_8U, cv::Scalar(0, 0, 0));
		cv::Mat ConbineTest;

		// 原本的變數
		int tmpIdx;
		int midIdx;
		float idt, idtSmooth, idtmax = 0, idtmin = 9999;
		int posZ;

		// Map 參數
		//QVector<IndexMapInfo> IndexMap;
		for (int row = 0; row < theTRcuda.VolumeSize_Y; row++)
		{
			// 這個 For 迴圈是算每一個結果
			// 也就是去算深度底下，1024 個結果
			for (int col = 1; col < theTRcuda.VolumeSize_Z; col++)
			{
				tmpIdx = ((x * theTRcuda.VolumeSize_Y) + row) * theTRcuda.VolumeSize_Z + col;
				midIdx = ((125 * theTRcuda.VolumeSize_Y) + row) * theTRcuda.VolumeSize_Z + col;
				idt = ((float)theTRcuda.VolumeData[tmpIdx] / (float)3.509173f) - (float)(3.39f / 3.509173f);// 調整後能量區間
				idtSmooth = ((float)theTRcuda.VolumeDataAvg[tmpIdx] / (float)3.509173f) - (float)(3.39f / 3.509173f);

				// 調整過飽和度的 顏色
				ImageResult.at<float>(row, col) = cv::saturate_cast<float>(1.5 * (idt - 0.5) + 0.5);
				SmoothResult.at<float>(row, col) = cv::saturate_cast<float>(1.5 * (idtSmooth - 0.5) + 0.5);
			}

			// 這個迴圈是去算
			tmpIdx = ((x * theTRcuda.VolumeSize_Y) + row) * theTRcuda.VolumeSize_Z;
			/*
			if (theTRcuda.PointType[tmpIdx + 3] == 1 && theTRcuda.PointType[tmpIdx + 1] != 0) 
			{
				posZ = theTRcuda.PointType[tmpIdx + 1];
				
				// 通常不會太靠近
				if(posZ  <  10)
					continue;

				IndexMapInfo tempInfo;
				tempInfo.index = row;
				tempInfo.ZValue = posZ;
				IndexMap.push_back(tempInfo);

				//cout << row << " " << posZ << endl;
				for (int i = posZ; i < 1024; i++)
					FastLabel.at<uchar>(row, i) = uchar(255);
			}*/
		}

		//////////////////////////////////////////////////////////////////////////
		// 找出邊界
		//////////////////////////////////////////////////////////////////////////
		//if (DoFastLabel && IndexMap.size() > 0)
		//{
		//	int arryIndex = 0;
		//	int FromIndex = 0;
		//	for (int row = 0; row < theTRcuda.VolumeSize_Y; row++)
		//	{
		//		// 這邊是正常情況下，剛剛好有找到的 Z 值
		//		if (IndexMap.size() <= arryIndex && arryIndex - 2 >= 0)
		//			FromIndex = LerpFunction(
		//				IndexMap[arryIndex - 2].index, IndexMap[arryIndex - 2].ZValue,
		//				IndexMap[arryIndex - 1].index, IndexMap[arryIndex - 1].ZValue,
		//				row
		//			);
		//		else if (row == IndexMap[arryIndex].index)
		//			FromIndex = IndexMap[arryIndex++].ZValue;
		//		else if (row < IndexMap[arryIndex].index && arryIndex > 0)
		//			// 因為第一個沒有 -1 ，所以需要額外分開來做
		//			FromIndex = LerpFunction(
		//				IndexMap[arryIndex - 1].index, IndexMap[arryIndex - 1].ZValue,
		//				IndexMap[arryIndex + 0].index, IndexMap[arryIndex + 0].ZValue,
		//				row
		//			);
		//		else  if (row < IndexMap[arryIndex].index && arryIndex == 0)
		//			FromIndex = LerpFunction(
		//				IndexMap[arryIndex + 0].index, IndexMap[arryIndex + 0].ZValue,
		//				IndexMap[arryIndex + 1].index, IndexMap[arryIndex + 1].ZValue,
		//				row
		//			);
		//		else
		//		{
		//			std::cout << "Fast Label 跳過 (" << row << ")!!" << std::endl;
		//			break;
		//		}
		//		FromIndex = qBound(0, FromIndex, theTRcuda.VolumeSize_Z - 1);
		//		// 填白色
		//		for (int i = FromIndex; i < theTRcuda.VolumeSize_Z; i++)
		//			FastLabel.at<uchar>(row, i) = uchar(255);
		//	}
		//}

		// 這邊是將 Contour & 結果，合再一起
		// 先複製一份
		ConbineTest = ImageResult.clone();
		ConbineTest.convertTo(ConbineTest, CV_8U, 255);
		cv::cvtColor(ConbineTest.clone(), ConbineTest, cv::COLOR_GRAY2BGR);
		for (int row = 0; row < theTRcuda.VolumeSize_Y; row++)
		{
			tmpIdx = ((x * theTRcuda.VolumeSize_Y) + row) * theTRcuda.VolumeSize_Z;
			if (theTRcuda.PointType[tmpIdx + 3] == 1 && theTRcuda.PointType[tmpIdx + 1] != 0)
			{
				posZ = theTRcuda.PointType[tmpIdx + 1];
				cv::Point contourPoint(posZ, row);
				cv::circle(ConbineTest, contourPoint, 1, cv::Scalar(0, 255, 255), CV_FILLED);
			}
		}

		// 使否只要顯示
		if (NeedSave)
		{
			//cv::resize(ImageResult, ImageResult, cv::Size(480, 360), 0, 0, cv::INTER_NEAREST);
			cv::Mat TempImage;
			ImageResult.convertTo(TempImage, CV_8U, 255);
			cv::imwrite("Images/OCTImages/origin_v2/" + std::to_string(x) + ".png", TempImage);

			SmoothResult.convertTo(TempImage, CV_8U, 255);
			cv::imwrite("Images/OCTImages/smooth_v2/" + std::to_string(x) + ".png", TempImage);

			//cv::resize(FastLabel.clone(), FastLabel, cv::Size(480, 360), 0, 0, cv::INTER_NEAREST);
			//cv::imwrite("Images/OCTImages/label_v2/" + std::to_string(x) + ".png", FastLabel);

			//cv::resize(ContourTest.clone(), ContourTest, cv::Size(480, 360), 0, 0, cv::INTER_NEAREST);
			cv::imwrite("Images/OCTImages/combine_v2/" + std::to_string(x) + ".png", ConbineTest);

		}

		// 暫存到陣列李
		ImageResultArray.push_back(ImageResult);
		SmoothResultArray.push_back(SmoothResult);
		//FastLabelArray.push_back(FastLabel);
		CombineTestArray.push_back(ConbineTest);
	}
	cout << "轉成圖片完成!!" << endl;
}
bool RawDataManager::ShakeDetect(QMainWindow *main, bool IsShowForDebug)
{
	// 錯誤判斷，判斷進入這個 Function 一定要有資料
	if (ImageResultArray.size() == 0)
	{
		QMessageBox::critical(main, codec->toUnicode("Function 錯誤"), codec->toUnicode("沒有資料可以執行!!"));
		return false;
	}
	cv::Mat FirstImage = SmoothResultArray[0];
	FirstImage.convertTo(FirstImage, CV_8U, 255);
	cv::Mat LastImage = SmoothResultArray[SmoothResultArray.size() - 1];
	LastImage.convertTo(LastImage, CV_8U, 255);

	// 這邊做兩個判斷
	// 1. 如果 PSNR 高於 30
	// 2. SURF 找出大於等於 2 個
	double PSNR_Value = PSNR(FirstImage, LastImage);
	cout << "PSNR: " << PSNR_Value << endl;
	if (PSNR_Value > 30)
		return true;
	return false;
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
