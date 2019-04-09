#pragma once
#include <QStringList>
#include <QVector>
#include <QFile>
#include <QIODevice>
#include <QVector3D>

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
typedef OpenMesh::TriMesh_ArrayKernelT<>  TriMesh;

using namespace std;
using namespace cv;

enum PredictDataType {
	EMPTY_TYPE = 0,			// 背景
	TEETH_TYPE,				// 牙齒
	MEAT_TYPE,				// 牙齦
	TONEBONE_TYPE			// 齒槽骨
};

class VolumeRenderClass
{
public:
	VolumeRenderClass(int, int, float*, float);
	~VolumeRenderClass();

	//////////////////////////////////////////////////////////////////////////
	// 外部呼叫函式
	//////////////////////////////////////////////////////////////////////////
	void ImportData(QString);													// 立體資料 (未來可以改成輸入預測結果，可能需要 BoundingBoxDataStruct)
	void SetRenderZ(int);														// 設定要畫的 Z (從 Widget 設定)
	//int GetRenderZ();															// 拿要畫得深度 (OpenGL 拿，怕衝突寫成 Get & Set)

	//////////////////////////////////////////////////////////////////////////
	// Render Buffer 相關
	//////////////////////////////////////////////////////////////////////////
	//void GenerateRenderBuffer();												// 產生 OpenGL 的繪畫的 Buffer (從哪一個 Z 開始建造)
	bool NeedReGenerateBuffer();												// 這邊是代表可以建立 Buffer 了
	void GenerateMeshFromLookZ();												// 拿從設定的 Z 中，往下看的部分
	QVector<QVector3D> Points;													// 點雲
	QVector<float> PointType;													// 點雲的 Type

private:
	//////////////////////////////////////////////////////////////////////////
	// 資料部分
	//////////////////////////////////////////////////////////////////////////
	PredictDataType* PredictType1D;												// 轉換到的資料
	int* InitFace;																// 初始面的深度
	int ImgWidth, ImgHeight;													// 圖片大小
	float* MappingMatrix;														// Mapping Data
	float zRatio;

	//////////////////////////////////////////////////////////////////////////
	// Render 相關
	//////////////////////////////////////////////////////////////////////////
	bool	NeedReGenerate = false;												// 是否要重新創建 Buffer
	int		ReGenerateStartZ = 0;												// 要從哪裡開始重建

	//////////////////////////////////////////////////////////////////////////
	// 參數設定
	//////////////////////////////////////////////////////////////////////////
	const int SizeYStart = 60;													// 起始點的張數
	const int SizeYEnd = 200;													// 結尾的張數

	//////////////////////////////////////////////////////////////////////////
	// Helper Function
	//////////////////////////////////////////////////////////////////////////
	int Get1DIndex(int, int, int);												// 拿索引值
	int GetMapping1D(int, int);													// 同上
	inline int clamp(int, int, int);											// clamp Function
};

