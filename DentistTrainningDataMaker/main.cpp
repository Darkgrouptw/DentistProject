/*
這邊是要從 Raw Data 產生資料

ConvertList.txt
檔案目錄
沒蓋肉 有蓋肉
沒蓋肉 有蓋肉
沒蓋肉 有蓋肉
.
.
.
*/
#include <iostream>

#include <QDir>
#include <QFile>
#include <QVector>
#include <QIODevice>

#include "RawDataManager.h"

using namespace std;

struct DataInfo
{
	QString UnCoveredPath;
	QString CoveredPath;
};

int main(int argc, char *argv[])
{
	#pragma region 變數宣告
	QVector<DataInfo *> DoList;
	RawDataManager dataM;
	#pragma endregion
	#pragma region 開檔 & 讀檔
	// 開檔
	QFile ConvertFile("ConvertList.txt");
	if (!ConvertFile.exists())
	{
		cout << "ConvertList 不存在" << endl;
		return -1;
	}

	// 讀檔
	ConvertFile.open(QIODevice::ReadOnly);
	QString Data = ConvertFile.readAll();
	Data = Data.replace("\r\n", "\n");
	cout << "讀取內容" << endl;
	cout << Data.toStdString() << endl;
	QStringList DataList = Data.split("\n");
	if (DataList.size() <= 1)
	{
		cout << "檔案內容不正確" << endl;
		return -1;
	}

	QDir ConvertDir(DataList[0]);
	for (int i = 1; i < DataList.size(); i++)
	{
		QStringList FileList = DataList[i].split(" ");
		DataInfo* info = new DataInfo();
		info->UnCoveredPath = ConvertDir.absoluteFilePath(FileList[0]);
		info->CoveredPath = ConvertDir.absoluteFilePath(FileList[1]);
		DoList.push_back(info);
	}
	#pragma endregion
	#pragma region 轉資料
	dataM.ReadRawDataFromFile(DoList[1]->CoveredPath);
	dataM.RawToPointCloud();
	dataM.TranformToIMG();
	#pragma endregion
	return 0;
}
