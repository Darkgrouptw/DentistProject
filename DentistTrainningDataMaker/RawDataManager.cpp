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


	//* // throw useless data
	int scanline = (DManager.rawDP.size_Z * 2) * DManager.rawDP.size_Y * 2;
	for (int i = scanline; i < buffer.size(); i += scanline)
		buffer.remove(i, scanline);

	std::cout << "size of buffer : " << buffer.size() << std::endl;
	DManager.rawDP.size_X = 2 * buffer.size() / ((DManager.rawDP.size_Z * 2) * DManager.rawDP.size_Y * 2);

	inputFile.close();
	t2 = clock();

	std::cout << "ReadRawData done t: " << (t2 - t1) / (double)(CLOCKS_PER_SEC) << " s" << std::endl;
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
		cv::Mat ContourTest; // = Mat(theTRcuda.VolumeSize_Y, theTRcuda.VolumeSize_Z, CV_8U, Scalar(0, 0, 0));

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
			//cout << row << " " << FromIndex << endl;
			

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
	/*if (nextValue >= 1024 || lastValue >= 1024 || nextValue < 0 || lastValue < 0)
		cout << "Error: " << nextValue << " " << lastValue << endl;
	cout << "Index: " << lastIndex << " " << nextIndex << " " << index << endl;*/
	return (index - lastIndex) * (nextValue - lastValue) / (nextIndex - lastIndex) + lastValue;
}
