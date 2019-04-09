#include "VolumeRenderClass.h"

VolumeRenderClass::VolumeRenderClass(int height, int width, float* MappingMatrix, float zRatios)
{
	ImgWidth = width;
	ImgHeight = height;
	PredictType1D = new PredictDataType[ImgWidth * ImgHeight * (SizeYEnd - SizeYStart + 1)];
	memset(PredictType1D, 0, sizeof(PredictDataType) * ImgWidth * ImgHeight * (SizeYEnd - SizeYStart + 1));
	InitFace = new int[ImgHeight *  (SizeYEnd - SizeYStart + 1)];

	this->MappingMatrix = MappingMatrix;
	zRatio = zRatios;
}
VolumeRenderClass::~VolumeRenderClass()
{
	delete[] PredictType1D;
	delete[] InitFace;
}

// 外部輸入
void VolumeRenderClass::ImportData(QString boundingBoxPath)
{
	#pragma region 把檔案讀進來
	QFile file(boundingBoxPath);
	assert(file.open(QIODevice::ReadOnly));

	// 先讀過一條
	file.readLine();

	QStringList lineData = QString(file.readAll()).split('\n');
	for (int i = SizeYStart; i <= SizeYEnd; i++)
	{
		QStringList content = lineData[i].split('\n')[0].split(" ");
		assert(content.size() >= 4);

		// 邊界
		int tlX = content[0].toInt();
		int tlY = content[1].toInt();
		int brX = content[2].toInt();
		int brY = content[3].toInt();
		
		Mat mat = imread("E:/DentistData/NetworkData/2019.01.08 ToothBone1/labeled_v2/" + to_string(i) + ".png");
		assert(mat.empty() == false && "缺少圖片!!");
		#pragma omp parallel for 
		for (int rowIndex = 0; rowIndex < mat.rows; rowIndex++)
			for (int colIndex = 0;colIndex < mat.cols; colIndex++)
			{
				Vec3b color = mat.at<Vec3b>(rowIndex, colIndex);
				int index = Get1DIndex(rowIndex + tlY, i, colIndex + tlX);
				if (color == Vec3b(0, 0, 0))
					PredictType1D[index] = PredictDataType::EMPTY_TYPE;
				else if (color == Vec3b(0, 0, 255))
					PredictType1D[index] = PredictDataType::TEETH_TYPE;
				else if (color == Vec3b(0, 255, 0))
					PredictType1D[index] = PredictDataType::MEAT_TYPE;
				else if (color == Vec3b(255, 0, 0))
					PredictType1D[index] = PredictDataType::TONEBONE_TYPE;
				else
				{
					cout << color << endl;
					assert(false && "錯誤!!");
				}
			}

	}
	#pragma endregion
	#pragma region 建造 Volume Data
	// 先找出第一個點
	for (int i = SizeYStart; i <= SizeYEnd; i++ )
		for (int j = 0; j < ImgHeight; j++)
		{
			int FirstZ = 0;
			for (;FirstZ < ImgWidth; FirstZ++)
			{
				if (PredictType1D[Get1DIndex(j, i, FirstZ)] != PredictDataType::EMPTY_TYPE)
					break;
			}

			// 判斷有沒有超過範圍
			if (FirstZ >= ImgWidth)
				FirstZ = -1;

			int index = (i - SizeYStart) * ImgHeight + j;
			InitFace[index] = FirstZ;
		}

	// 設定可以 重新建置 Buffer
	SetRenderZ(0);
	#pragma endregion
}
void VolumeRenderClass::SetRenderZ(int lookZ)
{
	// 一定要這個範圍內
	assert(0 <= lookZ && lookZ <= ImgWidth);

	// 先設定參數
	NeedReGenerate = true;
	ReGenerateStartZ = lookZ;
}


// Render Buffer 相關
bool VolumeRenderClass::NeedReGenerateBuffer()
{
	return NeedReGenerate;
}
void VolumeRenderClass::GenerateMeshFromLookZ()
{
	#pragma region 根據表面去往下找第一個非背景的點
	int KernalSize = 3;
	int* FirstFaceIndex = new int[ImgHeight *  (SizeYEnd - SizeYStart + 1)];
	for (int i = SizeYStart; i <= SizeYEnd; i++)
		for (int j = 0; j < ImgHeight; j++)
		{
			//int FirstZ = 0;
			//for (; FirstZ < ImgWidth; FirstZ++)
			//{
			//	if (PredictType1D[Get1DIndex(j, i, FirstZ)] != PredictDataType::EMPTY_TYPE)
			//		break;
			//}

			//// 判斷有沒有超過範圍
			//if (FirstZ >= ImgWidth)
			//	FirstZ = -1;

			int index = (i - SizeYStart) * ImgHeight + j;

			// 如果 Z 軸在 Init 的深度以下的話，就要繼續去往下找到是哪一類
			//if(ReGenerateStartZ > InitFace[index])

			//int VoteEmpty = 0;
			//int Vote
			int SmoothZ = InitFace[index];
			int VoteNum = 1;
			for (int yMoveIndex = -KernalSize; yMoveIndex <= KernalSize; yMoveIndex++)
				for (int xMoveIndex = -KernalSize; xMoveIndex <= KernalSize; xMoveIndex++)
				{
					int yIndex = clamp(i + yMoveIndex, SizeYStart, SizeYEnd);
					int xIndex = clamp(j + xMoveIndex, 0, ImgHeight);
					int moveIndex = (yIndex - SizeYStart) * ImgHeight + xIndex;
					SmoothZ += InitFace[moveIndex];
					VoteNum++;
				}
			FirstFaceIndex[index] = SmoothZ / VoteNum;
		}

	// Smooth
	#pragma endregion
	#pragma region 產生 Mesh
	// 清空點
	Points.clear();

	// 把點抓出來
	QVector3D MidPoint;
	for (int i = SizeYStart; i <= SizeYEnd - 1; i++)		// y(60 ~ 200)
		for (int j = 0; j < ImgHeight - 1; j++)				// x(0 ~ 249)
		{
			// 先判斷是否是有效的
			int index = (i - SizeYStart) * ImgHeight + j;
			int buttomIndex = (i - SizeYStart) * ImgHeight + (j + 1) ;
			int rightIndex = (i - SizeYStart + 1) * ImgHeight + j;
			int brIndex = (i - SizeYStart + 1) * ImgHeight + j + 1;
			if (FirstFaceIndex[index] != -1 && FirstFaceIndex[buttomIndex] != -1 && FirstFaceIndex[rightIndex] != -1 && FirstFaceIndex[brIndex] != -1)
			{
				//Points.push_back()
				float TLX = MappingMatrix[GetMapping1D(j, i) + 0];
				float TLY = MappingMatrix[GetMapping1D(j, i) + 1];
				float ButtomX = MappingMatrix[GetMapping1D(j + 1, i) + 0];
				float ButtomY = MappingMatrix[GetMapping1D(j + 1, i) + 1];
				float RightX = MappingMatrix[GetMapping1D(j, i + 1) + 0];
				float RightY = MappingMatrix[GetMapping1D(j, i + 1) + 1];
				float BRX = MappingMatrix[GetMapping1D(j + 1, i + 1) + 0];
				float BRY = MappingMatrix[GetMapping1D(j + 1, i + 1) + 1];

				float scaleZ = 2;
				QVector3D point =			QVector3D(-TLX, -TLY,				(float)FirstFaceIndex[index] / ImgWidth * scaleZ * zRatio);
				QVector3D buttomPoint =		QVector3D(-ButtomX, -ButtomY,		(float)FirstFaceIndex[buttomIndex] / ImgWidth * scaleZ * zRatio);
				QVector3D rightPoint =		QVector3D(-RightX, -RightY,			(float)FirstFaceIndex[rightIndex] / ImgWidth * scaleZ * zRatio);
				QVector3D brPoint =			QVector3D(-BRX, -BRY,				(float)FirstFaceIndex[brIndex] / ImgWidth * scaleZ * zRatio);

				Points.push_back(point);
				Points.push_back(buttomPoint);
				Points.push_back(rightPoint);
				MidPoint += point + buttomPoint + rightPoint;

				Points.push_back(buttomPoint);
				Points.push_back(brPoint);
				Points.push_back(rightPoint);
				MidPoint += buttomPoint + brPoint + rightPoint;

				// Point Type Smooth
				PredictDataType tempType = PredictType1D[Get1DIndex(j, i, FirstFaceIndex[index])];
				PointType.push_back((float)tempType);
				tempType = PredictType1D[Get1DIndex(j + 1, i, FirstFaceIndex[buttomIndex])];
				PointType.push_back((float)tempType);
				tempType = PredictType1D[Get1DIndex(j, i + 1, FirstFaceIndex[rightIndex])];
				PointType.push_back((float)tempType);

				tempType = PredictType1D[Get1DIndex(j + 1, i, FirstFaceIndex[buttomIndex])];
				PointType.push_back((float)tempType);
				tempType = PredictType1D[Get1DIndex(j + 1, i + 1, FirstFaceIndex[brIndex])];
				PointType.push_back((float)tempType);
				tempType = PredictType1D[Get1DIndex(j, i + 1, FirstFaceIndex[rightIndex])];
				PointType.push_back((float)tempType);
			}
		}

	// 位移
	MidPoint /= Points.size();
	MidPoint.setY(2 * MidPoint.y());
	for (int i = 0; i < Points.size(); i++)
		Points[i] -= MidPoint;

	// 測試 Mesh
	TriMesh mesh;
	for (int i = 0; i < Points.size(); i += 3)
	{
		TriMesh::VertexHandle vHandle[3];
		vHandle[0] = mesh.add_vertex(TriMesh::Point(Points[i + 0].x(), Points[i + 0].y(), Points[i + 0].z()));
		vHandle[1] = mesh.add_vertex(TriMesh::Point(Points[i + 1].x(), Points[i + 1].y(), Points[i + 1].z()));
		vHandle[2] = mesh.add_vertex(TriMesh::Point(Points[i + 2].x(), Points[i + 2].y(), Points[i + 2].z()));

		QVector<TriMesh::VertexHandle> face_vHandle;
		face_vHandle.push_back(vHandle[0]);
		face_vHandle.push_back(vHandle[1]);
		face_vHandle.push_back(vHandle[2]);
		mesh.add_face(face_vHandle.toStdVector());
	}
	OpenMesh::IO::write_mesh(mesh, "D:/a.obj");

	delete[] FirstFaceIndex;
	#pragma endregion
}

// Helper Function
int VolumeRenderClass::Get1DIndex(int x, int y, int z)
{
	return z +
		x * ImgWidth +
		(y - SizeYStart) * ImgWidth * ImgHeight;
}
int VolumeRenderClass::GetMapping1D(int x, int y)
{
	int MapID = (y * ImgHeight + x) * 2;	// 對應到 Mapping Matrix，在讀取的時候他是兩筆為一個單位
	return MapID;
}
int VolumeRenderClass::clamp(int n, int lower, int upper)
{
	return std::max(lower, std::min(n, upper - 1));
}