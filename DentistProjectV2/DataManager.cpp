#include "DataManager.h"

DataManager::DataManager(void)
{
	#pragma region 硬體參數設定
	Mapping_X = 250;
	Mapping_Y = 250;
	MappingMatrix = NULL;
	zRatio = 14.25822785;//5.72 * 2.5		// 這個後面會改
	#pragma endregion
	#pragma region 傳入參數設定
	prop.SizeX				= 250;
	prop.SizeY				= 250;
	prop.SizeZ				= 2048;
	prop.ShiftValue			= 37 * 4 - 4;
	prop.K_Step				= 2;
	prop.CutValue			= 10;
	#pragma endregion

}
DataManager::~DataManager(void)
{
	delete[] MappingMatrix;
}

// 讀取校正檔
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
