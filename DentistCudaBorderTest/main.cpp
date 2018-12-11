#include "CudaBorder.cuh"

#include <iostream>
#include <cassert>
#include <Windows.h>

#include <QVector>
#include <QFile>
#include <QIODevice>

using namespace std;

// Data
float*  _FileData = NULL;
float** FileData = NULL;
int sizeY, sizeZ;
CudaBorder cuda;

// Helper Function
void SaveDelete(void *pointer)
{
	if (pointer != NULL)
		delete pointer;
}

// 讀取檔案
void ReadFile(QString FileLocation)
{
	// 計時
	clock_t time = clock();

	// 檔案位置
	QFile file(FileLocation);
	assert(file.open(QIODevice::ReadOnly));

	QString string = file.readAll();
	QStringList stringList = string.split("\n");
	QStringList FirstLine = stringList[0].split("\t");

	sizeY = stringList.count() - 1;									// pass 掉最後一個 \n
	sizeZ = FirstLine.count() - 1 + 1;								// pass 掉最後一個 \t (最補上最前面的 0)

	// Mapping Data 到 Array
	_FileData = new float[sizeY * sizeZ];
	FileData = new float*[sizeY];
	for (int i = 0; i < sizeY; i++)
	{
		// 分開 \t
		FirstLine = stringList[i].split("\t");

		// 將資料放進去
		FileData[i] = &_FileData[i * sizeZ];
		_FileData[i * sizeZ] = 0;									// 一開始是 0
		for (int j = 1; j < sizeZ; j++)
			_FileData[i * sizeZ + j] = FirstLine[j].toFloat();
	}
	
	// 關閉檔案
	file.close();

	// 時間
	time = clock() - time;
	cout << "Pass 資料: " << ((float)time) / CLOCKS_PER_SEC << " sec" << endl;
}

void DeleteData()
{
	SaveDelete((float*)FileData);
	SaveDelete(_FileData);
}
int main(int argc, char *argv[])
{
	// 讀取檔案
	//ReadFile("D:/Dentist/Data/ScanData/2018.11.28/1_raw/60.txt");
	//ReadFile("D:/Dentist/Data/ScanData/2018.11.28/3_raw/60.txt");
	//ReadFile("D:/Dentist/Data/ScanData/2018.10.18/20181016_Incisor_rawdata/120.txt");
	//ReadFile("C:/Users/Dark/Desktop/SourceTree/DentistProject/x64/Release/DentistProject/Images/OCTImages/rawdata_v2/60.txt");
	//ReadFile("C:/Users/Dark/Desktop/SourceTree/DentistProject/x64/Release/DentistProject/Images/OCTImages/rawdata_v2/120.txt");

	// 抓出邊界 
	cuda.Init(sizeY, sizeZ);
	cuda.GetBorderFromCuda(FileData);

	// 轉換成圖檔
	QImage img = cuda.SaveDataToImage(FileData);
	img.save("D:/a.png");

	DeleteData();
	system("PAUSE");
	return 0;
}
