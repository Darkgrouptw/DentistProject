#include "TRCudaV2.cuh"

#include <iostream>
#include <Windows.h>

#include <QDataStream>
#include <QTextStream>
#include <QFile>
#include <QString>

using namespace std;

int main(int argc, char *argv[])
{
	#pragma region 例外狀況排除
	// Windows 10 1809 的 bug
	system("chcp 65001");
	system("chcp 950");

	if (argc != 2)
	{
		cout << "檔名錯誤" << endl;
		return -1;
	}

	QString FileName = QString(argv[1]);
	QFile inputFile(FileName);
	cout << FileName.toLocal8Bit().constData() << endl;
	if (!inputFile.open(QIODevice::ReadOnly))
	{
		cout << "Raw Data 讀取錯誤" << endl;
		return -1;
	}
	else
		cout << "讀取 Raw Data: " << FileName.toLocal8Bit().constData() << endl;

	#pragma endregion
	#pragma region 抓取全部的 Bytes
	int bufferSize = inputFile.size() / sizeof(quint8);

	QDataStream qData(&inputFile);
	QByteArray buffer;
	buffer.clear();
	buffer.resize(bufferSize);
	int status = qData.readRawData(buffer.data(), bufferSize);

	//-1 代表有錯誤
	assert(status != -1);

	inputFile.close();
	#pragma endregion
	#pragma region 開始做轉換
	// 讀資料 (後面的參數是東元那邊測試出來的 )
	TRCudaV2 cudaV2;
	cudaV2.RawDataToPointCloud(buffer.data(), bufferSize, 250, 250, 2048, 37 * 4 - 4, 2, 10);
	#pragma endregion
	#pragma region 測試 Part
	// 檔案測試
	QFile testFile("testFile.txt");
	QString content;
	testFile.open(QIODevice::WriteOnly);

	QTextStream ss(&testFile);
	// 讀檔 正掃
	for (int i = 0; i < 4096; i++)
		content += QString::number(cudaV2.OCTData[i]) + "\n";
	// 讀檔 反掃
	/*for (int i = 2048 * 250 * 3; i < 2048 * 250 * 3 + 4096; i++)
		content += QString::number(cudaV2.OCTData[i]) + "\n";
	*/

	// 確保轉換完 ushort 後，還有資料
	/*for (int i = bufferSize / 2 - 1024; i < bufferSize / 2; i++)
		content += QString::number(cudaV2.OCTData[i]) + "\n";*/
	ss << content;
	testFile.close();
	#pragma endregion
	system("PAUSE");
	return 0;
}
