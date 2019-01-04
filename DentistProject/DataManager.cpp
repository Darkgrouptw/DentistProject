#include "DataManager.h"

DataManager::DataManager(void)
{
	// variable init
	rawDP.size_X = 250;
	rawDP.size_Y = 250;
	rawDP.size_Z = 2048; // 1376 1600
	rawDP.sample = 1024;

	Mapping_X = 250;
	Mapping_Y = 250;
	MappingMatrix = NULL;
	zRatio = 14.25822785;//5.72 * 2.5
}
DataManager::~DataManager(void)
{
	delete[] MappingMatrix;
}

//////////////////////////////////////////////////////////////////////////
// 讀取校正檔
//////////////////////////////////////////////////////////////////////////
void DataManager::ReadCalibrationData()
{
	std::ifstream myfile;
	myfile.open("Mapping20170818.txt");
	if (!myfile.is_open()) {
		std::cout << "failed to open!" << std::endl;
	}
	double number = 0;
	std::vector<double> numberlist;
	int MappingSize = Mapping_X * Mapping_Y;
	MappingMatrix = new float[MappingSize * 2];
	while (myfile >> number) {
		numberlist.push_back(number);
	}
	//std::cout << "number size:" << numberlist.size() << std::endl;
	for (int i = 0; i < MappingSize; i++) {
		MappingMatrix[i * 2] = numberlist[i * 4 + 2];
		MappingMatrix[i * 2 + 1] = numberlist[i * 4 + 3];
	}
	std::cout << "讀完 Mapping Data" << std::endl;
	myfile.close();
}
void DataManager::Read_test_file()
{
	char buffer[256];
	long size;
	std::ifstream myfile;
	myfile.open("Test_file", std::fstream::binary);
	if (!myfile.is_open()) {
		std::cout << "failed to open!" << std::endl;
	}
}