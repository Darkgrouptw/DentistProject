#include <QtCore/QCoreApplication>
#include <iostream>
#include <windows.h>
#include <msclr\marshal_cppstd.h>
//#include <QDir>

using namespace std;

int main(int argc, char *argv[])
{
	unsigned short IntData = 6;
	unsigned short OutData = 0;
	cli::array<unsigned short> ^IntArray;
	System::String ^TestString;

	//cout << "現在目錄：" << QDir::currentPath().toStdString() << endl;
	OCTTest::OCTTest::t1(IntData, OutData, IntArray, TestString);
	cout << IntData << endl;
	cout << OutData << endl;

	for (int i = 0; i< IntArray->Length; i++)
	{
		cout << IntArray[i] << " ";
	}
	cout << endl;
	std::string unmanaged = msclr::interop::marshal_as<std::string>(TestString);
	cout << unmanaged << endl;
	system("PAUSE");
	return 0;
}
