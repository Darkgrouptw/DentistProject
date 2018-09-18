#include <QtCore/QCoreApplication>
#include <QFile>
#include <QDir>

#include <iostream>

using namespace std;

void MoveUselessFile(QDir JobPath)
{
	#pragma region 判斷資料夾在不在
	cout << "目錄：" << JobPath.path().toStdString() << endl;
	if (!JobPath.exists())
	{
		cout << "無此目錄" << endl;
		return;
	}
	#pragma endregion
	#pragma region 新增 UnTraining 的目錄
	QStringList list = JobPath.path().split("/");
	QString DirectoryName = "UnTrainning_" + list[list.count() - 1];
	QDir UnTrainDir(JobPath.path() + "/../" + DirectoryName);
	cout << "產生目錄：" << UnTrainDir.absolutePath().toStdString() << endl;
	if (!UnTrainDir.exists())
		UnTrainDir.mkdir(".");
	#pragma endregion
	#pragma region 把東西搬過去
	for (int i = 0; i <= 59; i++)
		JobPath.rename("./" + QString::number(i) + ".png", UnTrainDir.absoluteFilePath(QString::number(i) + ".png"));
	for (int i = 201; i <= 249; i++)
		JobPath.rename("./" + QString::number(i) + ".png", UnTrainDir.absoluteFilePath(QString::number(i) + ".png"));
	#pragma endregion
}

int main(int argc, char *argv[])
{
	QCoreApplication a(argc, argv);

	if (argc != 2)
	{
		cout << "使用方式：<exe> <資料夾目錄>" << endl;
		cout << "EX: DentistDataCommand.exe ./UnCover_1" << endl;
		return 0;
	}

	// 這部分是要把資料夾裡面，把其他沒有用的 0~ 59 & 201 ~ 249
	//cout << argv[1] << endl;
	MoveUselessFile(QDir(argv[1]));
	cout << "完成!!" << endl;
	return 0;
}
