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
		cout << "Raw Data 讀取錯誤" << endl;
		return;
	}

	int bufferSize = inputFile.size() / sizeof(quint8);
	cout << "Size : " << bufferSize << endl;

	QDataStream qData(&inputFile);
	buffer.clear();
	buffer.resize(bufferSize);
	qData.readRawData(buffer.data(), bufferSize);


	//* // throw useless data
	int scanline = (DManager.rawDP.size_Z * 2) * DManager.rawDP.size_Y * 2;
	for (int i = scanline; i < buffer.size(); i += scanline)
		buffer.remove(i, scanline);

	cout << "size of buffer : " << buffer.size() << endl;
	DManager.rawDP.size_X = 2 * buffer.size() / ((DManager.rawDP.size_Z * 2) * DManager.rawDP.size_Y * 2);

	inputFile.close();
	t2 = clock();

	cout << "ReadRawData done t: " << (t2 - t1) / (double)(CLOCKS_PER_SEC) << " s" << endl;
}

void RawDataManager::RawToPointCloud()
{
	RawDataProperty *tmpRDP = &DManager.rawDP;
	theTRcuda.RawToPointCloud(buffer.data(), buffer.size(), tmpRDP->size_Y, tmpRDP->size_Z, 1);
}

void RawDataManager::TranformToIMG()
{
	int tmpIdx;
	int midIdx;
	float idt, idtmax = 0, idtmin = 9999;
	float ori_idt, total_idt = 0, avg_idt = 0;
	fstream fp;
	fp.open("vol_data.txt", ios::out);


	for (int x = 0; x < theTRcuda.VolumeSize_X; x++) {
		cv::Mat tmp_floating = cv::Mat(theTRcuda.VolumeSize_Y, theTRcuda.VolumeSize_Z, CV_32F, cv::Scalar(0, 0, 0));
		cv::Mat tmp_test = cv::Mat(theTRcuda.VolumeSize_Y, theTRcuda.VolumeSize_Z, CV_32F, cv::Scalar(0, 0, 0));
		for (int row = 0; row < theTRcuda.VolumeSize_Y; row++) {
			for (int col = 0; col < theTRcuda.VolumeSize_Z; col++) {
				tmpIdx = ((x * theTRcuda.VolumeSize_Y) + row) * theTRcuda.VolumeSize_Z + col;
				midIdx = ((125 * theTRcuda.VolumeSize_Y) + row) * theTRcuda.VolumeSize_Z + col;

				ori_idt = (float)theTRcuda.VolumeData[tmpIdx];
				idt = ((float)theTRcuda.VolumeDataAvg[tmpIdx] / (float)3.509173f) - (float)(3.39f / 3.509173f);// 調整後能量區間

				tmp_floating.at<float>(row, col) = idt;
				tmp_test.at<float>(row, col) = (float)(ori_idt - 1) / 8;
				if (ori_idt > idtmax)
					idtmax = ori_idt;
				if (ori_idt < idtmin)
					idtmin = ori_idt;

				total_idt += ori_idt;
			}
		}

		cv::resize(tmp_floating, tmp_floating, cv::Size(480, 360), 0, 0, CV_INTER_LINEAR);
		tmp_floating.convertTo(tmp_floating, CV_8UC3, 255);

		cv::imwrite("origin_v2\\" + std::to_string(x) + ".png", tmp_floating);

	}
	fp.close();
}
