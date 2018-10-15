#include "RawDataManager.h"

RawDataManager::RawDataManager()
{
	DManager.ReadCalibrationData();
}
RawDataManager::~RawDataManager()
{
}

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
}
void RawDataManager::ScanDataFromDevice()
{
	/*
	pin_ptr<int32_t> tmp_deviceID = &deviceID;
	InitADC(4, tmp_deviceID);
	clock_t scan_t1 = clock();

	//AboutADC(deviceID);
	/////////////////////////////////////////////////////////H_StartCap///////////////////////////////////////////////
	float LV_65 = 65;
	char SaveName[] = "V:\\OCT20170928";
	unsigned int SampRec = 2048;

	char SaveName2[1024];
	std::string name_title = "V:\\";

	(*out_fileName) = name_title + (*out_fileName);

	strcpy(SaveName2, out_fileName->c_str());

	#ifndef TEST_NO_OCT
	SerialPort port("COM6", 9600);
	port.Open();

	if (!port.IsOpen) {
		std::cout << "COM6 fail to open!" << std::endl;
	}
	#endif // !TEST_NO_OCT
	Sleep(100);

	Savedata = true;
	ErrorBoolean = false;
	ByteLen = 1;


	pin_ptr<int32_t> tmp_ErrorString_len_out = &ErrorString_len_out;
	pin_ptr<uint32_t> tmp_Handle = &HandleOut;
	pin_ptr<uint32_t> tmp_ByteLen = &ByteLen;
	pin_ptr<LVBoolean> tmp_ErrorBoolean = &ErrorBoolean;


#ifndef TEST_NO_OCT
	StartCap(deviceID, tmp_Handle, LV_65, SampRec, tmp_ByteLen, Savedata, SaveName, tmp_ErrorBoolean, ErrorString, ErrorString_len_in, tmp_ErrorString_len_out);
	port.RtsEnable = true;
#else
	test_start_cap(deviceID, tmp_Handle, LV_65, SampRec, tmp_ByteLen, Savedata, SaveName2, tmp_ErrorBoolean, ErrorString, ErrorString_len_in, tmp_ErrorString_len_out);
#endif // !TEST_NO_OCT



	/////////////////////////////////////////////////////////H_StartCap///////////////////////////////////////////////


	/////////////////////////////////////////////////////////H_ScanADC///////////////////////////////////////////////

	AllDatabyte = ByteLen * 2;
	ArrSize = new unsigned short[PIC_SIZE];
	outarr = new unsigned short[PIC_SIZE];
	final_oct_arr = new unsigned short[PIC_SIZE * 125];
	final_oct_char = new char[PIC_SIZE * 250];

	OutArrLenIn = 2048 * 500 * 2;
	OutArrLenOut = 0;
	//button_stop_enable = true;
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
#ifdef TEST_NO_OCT
		test_scanADC(HandleOut, AllDatabyte, ArrSize, ByteLen, outarr, OutArrLenIn, tmp_OutArrLenOut, ErrorString, ErrorString_len_in, tmp_ErrorString_len_out);
#else
		ScanADC(HandleOut, AllDatabyte, ArrSize, ByteLen, outarr, OutArrLenIn, tmp_OutArrLenOut, ErrorString, ErrorString_len_in, tmp_ErrorString_len_out);
#endif // TEST_NO_OCT

		memcpy(final_oct_arr, outarr, sizeof(unsigned short) * PIC_SIZE);

		pic_count++;

		//std::cout << "pic_count" << pic_count << std::endl;
		//if(pic_count!= 249)

		final_oct_arr = final_oct_arr + PIC_SIZE;

		//if (pic_count == 63&& !change_point_cloud) {
		//	final_oct_arr = interator;
		//	change_point_cloud = true;
		//}


	}
	#ifndef TEST_NO_OCT
	AboutADC(deviceID);
	port.RtsEnable = false;
	port.Close();
	//std::cout << "tmp_char_finish " << std::endl;
	#endif // !TEST_NO_OCT

	t4 = clock();

	unsigned_short_to_char(interator, PIC_SIZE * 125, final_oct_char);
	t5 = clock();

	std::vector<char> tmp_char(final_oct_char, final_oct_char + PIC_SIZE * 250);
	//memccpy(tmp_char.data(), final_oct_char,std::strlen(final_oct_char), sizeof(char));

	std::cout << "to array time:" << (t4 - t3) / (double)(CLOCKS_PER_SEC) << "s" << std::endl;
	std::cout << "short to char time:" << (t5 - t4) / (double)(CLOCKS_PER_SEC) << "s" << std::endl;

	//int scanline = (2048 * 2) * 250 * 2;
	//for (int i = tmp_char.size() - scanline - 1; i > 0; i -= scanline * 2)
	//{
	//	tmp_char.erase(tmp_char.begin() + i, tmp_char.begin() + i + scanline);
	//}
	//if (theTRcuda != NULL) {
	//	delete theTRcuda;
	//	theTRcuda = new TRcuda;
	//}

	theTRcuda->RawToPointCloud(tmp_char.data(), tmp_char.size(), 250, 2048);
	//std::cout << "Scan3D!" << std::endl;
	scan_count++;

	int PCidx, Mapidx;
	float ratio = 1;
	float zRatio = DManager->zRatio;
	PointCloudArray tmpPC_T;
	//std::vector<PointData> tmpPC;
	//std::vector<int> tmp_VolumeData;

	for (int x = 2; x < theTRcuda->VolumeSize_X - 2; x++)
	{
		for (int y = 0; y < theTRcuda->VolumeSize_Y; y++)
		{
			if (x > DManager->Mapping_X || y > DManager->Mapping_Y)
				continue;
			//Mapidx = ((y)* theTRcuda->VolumeSize_X) + x;
			Mapidx = ((y * theTRcuda->sample_Y * theTRcuda->VolumeSize_X) + x) * theTRcuda->sample_X;
			PCidx = ((x * theTRcuda->VolumeSize_Y) + y) * theTRcuda->VolumeSize_Z;
			if (theTRcuda->PointType[PCidx + 3] == 1 && theTRcuda->PointType[PCidx + 1] != 0)
			{
				GlobalRegistration::Point3D tmp;
				PointData tmp_Data;
				//int tmpVolumeData;
				//tmp.x() = DManager->MappingMatrix[Mapidx * 2] * ratio + 0.2;
				//tmp.y() = DManager->MappingMatrix[Mapidx * 2 + 1] * ratio;
				//tmp.z() = theTRcuda->PointType[PCidx + 1] * zRatio / theTRcuda->VolumeSize_Z * ratio;
				tmp_Data.Position.x() = DManager->MappingMatrix[Mapidx * 2 + 1] * ratio + 0.2;
				tmp_Data.Position.y() = DManager->MappingMatrix[Mapidx * 2] * ratio;
				tmp_Data.Position.z() = theTRcuda->PointType[PCidx + 1] * zRatio / theTRcuda->VolumeSize_Z * ratio;
				//tmp_VolumeData.push_back(theTRcuda->VolumeData[PCidx]);
				//tmpPC_T.mPC.
				tmpPC_T.mPC.push_back(tmp_Data);
				if (x == 124) {
					point_z_124 = point_z_124 + tmp_Data.Position.z();
				}
				if (x == 125) {
					point_z_125 = point_z_125 + tmp_Data.Position.z();
				}
				//tmpPC.push_back(tmp);
			}
		}
	}

	if (tmpPC_T.mPC.size() > 10000) {
		(*PointCloudArr).push_back(tmpPC_T);
		//(*pointCloudSet).push_back(tmpPC);
		PointCloud_idx_show = (*PointCloudArr).size() - 1;

		(*PointCloudArr)[PointCloud_idx_show].mPC_noGyro = tmpPC_T.mPC;
		std::cout << "tmpPC_T.mPC.size(): " << tmpPC_T.mPC.size() << std::endl;
		//std::cout << "(*PointCloudArr)[PointCloud_idx_show].mPC_noGyro.size(): " << (*PointCloudArr)[PointCloud_idx_show].mPC_noGyro.size() << std::endl;
		///////////////////////////////
		//if (pointCloudSet->size() == 1) {
		//	CombinePC(0);
		//}

		std::cout << "point_z_124 Total: " << point_z_124 << std::endl;
		std::cout << "point_z_125 Total: " << point_z_125 << std::endl;

		Find_max_min();
		draw_before_mapping();
		//numericUpDown4->Maximum = (*PointCloudArr).size() - 1;
		std::cout << "now point cloud size:" << (*PointCloudArr).size() << std::endl;
		std::cout << "Scan count:" << scan_count << std::endl;
	}

	//std::cout << "PointCloud_idx_show: " << PointCloud_idx_show << std::endl;

	is_camera_move = true;

	//if ((*PointCloudArr).size() == 1) {
	//	//*quat1 = sysManager->GetGloveDataLPtr()->GetQuaternion();
	//	//*quat1 = quat1->Inverse();
	//	//std::cout << "quat1" << (*quat1) << "\n";


	//	//*acc_vector1 = sysManager->GetGloveDataLPtr()->GetAcceleration();
	//	first_t = clock();

	//	//Radian *tmp_diff;
	//	//Vector3 *tmp_rotation;
	//	//std::vector<GlobalRegistration::Point3D> *tmp_PC;
	//	//tmp_PC = new std::vector<GlobalRegistration::Point3D>;
	//	int tmp_nearest = 0;
	//	//tmp_diff = new Radian;
	//	//tmp_rotation = new Vector3;
	//	//(*pointCloud_radian).push_back(*tmp_diff);
	//	//(*pointCloud_quat_vector).push_back(*tmp_rotation);
	//	//(*nearest_idx).push_back(tmp_nearest);
	//	//(*nearest_idx).push_back(tmp_nearest);
	//	//(*pointCloud_aligned).push_back(*tmp_PC);

	//	//(*cloud_vec_arr).push_back(*tmp_rotation);
	//	//(*near_idx2).push_back(tmp_nearest);
	//	//(*near_idx2).push_back(tmp_nearest);
	//}
	//else if ((*PointCloudArr).size() == 2) {
	//	Rotate_cloud();
	//	std::cout << "now idx:" << PointCloud_idx_show << "     now size is:" << (*PointCloudArr).size() << "   mPC size is:" << (*PointCloudArr)[PointCloud_idx_show].mPC.size() << "   degree is:" << (*PointCloudArr)[PointCloud_idx_show].cloud_degree_arr;
	//}
	//else if ((*PointCloudArr).size()>2) {
	//	Rotate_cloud();
	//	Find_min_quat3();
	//	std::cout << "now idx:" << PointCloud_idx_show << "     now size is:" << (*PointCloudArr).size() << "   mPC size is:" << (*PointCloudArr)[PointCloud_idx_show].mPC.size() << "   degree is:" << (*PointCloudArr)[PointCloud_idx_show].cloud_degree_arr;
	//}
	//if ((*PointCloudArr).size() > 1) {
	//	Point_cloud_idx_sec = (*pointCloudSet).size() - 2;
	//}
	//clear_PC_buffer();
	//color_img();

	delete[] interator;
	//delete[] final_oct_arr;
	delete[] final_oct_char;
	delete[] ArrSize;
	delete[] outarr;

	clock_t scan_t2 = clock();
	full_scan_time = (scan_t2 - scan_t1) / (double)(CLOCKS_PER_SEC);



	all_time = all_time + full_scan_time;

	std::cout << "full_scan_time:" << full_scan_time << "s" << std::endl;
	std::cout << "all_time:" << all_time << "s" << std::endl;
	*/
}

void RawDataManager::RawToPointCloud()
{
	RawDataProperty *tmpRDP = &DManager.rawDP;
	theTRcuda.RawToPointCloud(buffer.data(), buffer.size(), tmpRDP->size_Y, tmpRDP->size_Z, 1);
}

void RawDataManager::TranformToIMG()
{
	//////////////////////////////////////////////////////////////////////////
	// 這邊底下是舊的 Code
	// 不是我寫的 QAQ
	//////////////////////////////////////////////////////////////////////////
	// 取 60 ~ 200
	//for (int x = 0; x < theTRcuda.VolumeSize_X; x++) 
	for (int x = 60; x <= 200; x++)
	{
		// Mat
		cv::Mat ImageResult = cv::Mat(theTRcuda.VolumeSize_Y, theTRcuda.VolumeSize_Z, CV_32F, cv::Scalar(0, 0, 0));
		cv::Mat FastLabel = cv::Mat(theTRcuda.VolumeSize_Y, theTRcuda.VolumeSize_Z, CV_8U, cv::Scalar(0, 0, 0));
		cv::Mat ContourTest;

		// 原本的變數
		int tmpIdx;
		int midIdx;
		float idt, idtmax = 0, idtmin = 9999;
		int posZ;

		// Map 參數
		QVector<IndexMapInfo> IndexMap;
		for (int row = 0; row < theTRcuda.VolumeSize_Y; row++)
		{
			// 這個 For 迴圈是算每一個結果
			// 也就是去算深度底下，1024 個結果
			for (int col = 0; col < theTRcuda.VolumeSize_Z; col++)
			{
				tmpIdx = ((x * theTRcuda.VolumeSize_Y) + row) * theTRcuda.VolumeSize_Z + col;
				midIdx = ((125 * theTRcuda.VolumeSize_Y) + row) * theTRcuda.VolumeSize_Z + col;
				idt = ((float)theTRcuda.VolumeData[tmpIdx] / (float)3.509173f) - (float)(3.39f / 3.509173f);// 調整後能量區間

				// 調整過飽和度的 顏色
				ImageResult.at<float>(row, col) = cv::saturate_cast<float>(1.5 * (idt - 0.5) + 0.5);
				//ImageResult.at<float>(row, col) = idt;
			}

			// 這個迴圈是去算
			tmpIdx = ((x * theTRcuda.VolumeSize_Y) + row) * theTRcuda.VolumeSize_Z;
			if (theTRcuda.PointType[tmpIdx + 3] == 1 && theTRcuda.PointType[tmpIdx + 1] != 0) 
			{
				posZ = theTRcuda.PointType[tmpIdx + 1];
				
				// 通常不會太靠近
				if(posZ  <  100)
					continue;

				IndexMapInfo tempInfo;
				tempInfo.index = row;
				tempInfo.ZValue = posZ;
				IndexMap.push_back(tempInfo);

				//cout << row << " " << posZ << endl;
				for (int i = posZ; i < 1024; i++)
					FastLabel.at<uchar>(row, i) = uchar(255);
			}
		}

		//////////////////////////////////////////////////////////////////////////
		// 找出邊界
		//////////////////////////////////////////////////////////////////////////
		int arryIndex = 0;
		int FromIndex = 0;
		for (int row = 0; row < theTRcuda.VolumeSize_Y; row++)
		{
			// 這邊是正常情況下，剛剛好有找到的 Z 值
			if (IndexMap.size() <= arryIndex && arryIndex - 2 >= 0)
				FromIndex = LerpFunction(
					IndexMap[arryIndex - 2].index, IndexMap[arryIndex - 2].ZValue,
					IndexMap[arryIndex - 1].index, IndexMap[arryIndex - 1].ZValue,
					row
				);
			else if (row == IndexMap[arryIndex].index)
				FromIndex = IndexMap[arryIndex++].ZValue;
			else if (row < IndexMap[arryIndex].index && arryIndex > 0)
				// 因為第一個沒有 -1 ，所以需要額外分開來做
				FromIndex = LerpFunction(
					IndexMap[arryIndex - 1].index, IndexMap[arryIndex - 1].ZValue,
					IndexMap[arryIndex + 0].index, IndexMap[arryIndex + 0].ZValue,
					row
				);
			else  if (row < IndexMap[arryIndex].index && arryIndex == 0)
				FromIndex = LerpFunction(
					IndexMap[arryIndex + 0].index, IndexMap[arryIndex + 0].ZValue,
					IndexMap[arryIndex + 1].index, IndexMap[arryIndex + 1].ZValue,
					row
				);
			else
				std::cout << "Error!!" << std::endl;
			FromIndex = qBound(0, FromIndex, theTRcuda.VolumeSize_Z - 1);
			

			// 填白色
			for (int i = FromIndex; i < theTRcuda.VolumeSize_Z; i++)
				FastLabel.at<uchar>(row, i) = uchar(255);
		}

		// 這邊是將 Contour & 結果，合再一起
		// 先複製一份
		ContourTest = ImageResult.clone();
		ContourTest.convertTo(ContourTest, CV_8U, 255);
		cv::cvtColor(ContourTest.clone(), ContourTest, cv::COLOR_GRAY2BGR);
		for (int row = 0; row < theTRcuda.VolumeSize_Y; row++)
		{
			tmpIdx = ((x * theTRcuda.VolumeSize_Y) + row) * theTRcuda.VolumeSize_Z;
			if (theTRcuda.PointType[tmpIdx + 3] == 1 && theTRcuda.PointType[tmpIdx + 1] != 0)
			{
				posZ = theTRcuda.PointType[tmpIdx + 1];
				cv::Point contourPoint(posZ, row);
				cv::circle(ContourTest, contourPoint, 1, cv::Scalar(0, 255, 255), CV_FILLED);
			}
		}

		cv::resize(ImageResult, ImageResult, cv::Size(480, 360), 0, 0, cv::INTER_NEAREST);
		ImageResult.convertTo(ImageResult, CV_8U, 255);
		cv::imwrite("origin_v2/" + std::to_string(x) + ".png", ImageResult);

		cv::resize(FastLabel.clone(), FastLabel, cv::Size(480, 360), 0, 0, cv::INTER_NEAREST);
		cv::imwrite("label_v2/" + std::to_string(x) + ".png", FastLabel);

		cv::resize(ContourTest.clone(), ContourTest, cv::Size(480, 360), 0, 0, cv::INTER_NEAREST);
		cv::imwrite("combine_v2/" + std::to_string(x) + ".png", ContourTest);
	}
}

int RawDataManager::LerpFunction(int lastIndex, int lastValue, int nextIndex, int nextValue, int index)
{
	return (index - lastIndex) * (nextValue - lastValue) / (nextIndex - lastIndex) + lastValue;
}
