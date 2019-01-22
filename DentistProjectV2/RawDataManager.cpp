#include "RawDataManager.h"

RawDataManager::RawDataManager()
{
	cout << "OpenCV Version: " << CV_VERSION << endl;
	DManager.ReadCalibrationData();
}
RawDataManager::~RawDataManager()
{
}

// UI 相關
void RawDataManager::SendUIPointer(QVector<QObject*> UIPointer)
{
	// 確認是不是有多傳，忘了改的
	assert(UIPointer.size() == 3);
	ImageResult			= (QLabel*)UIPointer[0];
	NetworkResult		= (QLabel*)UIPointer[1];
	FinalResult			= (QLabel*)UIPointer[2];
}
void RawDataManager::ShowImageIndex(int index)
{
	// 這邊有兩種狀況
	// 1. 第一種狀況是 60 ~ 200
	// 2. 第二種狀況是 只有掃描一張的結果
	if (60 <= index && index <= 200 && QImageResultArray.size() > 0)
	{
		//QImage Pixmap_ImageResult = QImageResultArray[index];
		//ImageResult->setPixmap(QPixmap::fromImage(Pixmap_ImageResult));

		////QImage Pixmap_NetworkResult = Mat2QImage(FastBorderResultArray[index - 60], CV_8UC3);
		////NetworkResult->setPixmap(QPixmap::fromImage(Pixmap_NetworkResult));

		//// 如果有東西的話才顯示 Network 預測的結果
		//if (QCombineResultArray.size() > 0)
		//{
		//	QImage Pixmap_FinalResult = QCombineResultArray[index];
		//	FinalResult->setPixmap(QPixmap::fromImage(Pixmap_FinalResult));
		//}
	}
}

// OCT 相關的步驟
void RawDataManager::Scan_SingleData_FromDeviceV2(QString SaveFileName, bool NeedSave_RawData)
{
	#pragma region 初始化裝置
	OCT64::OCT64::Init(
		4,
		OCT_DeviceID
	);
	#pragma endregion
	#pragma region 開 Port
	SerialPort port(gcnew System::String(OCTDevicePort.c_str()), 9600);
	port.Open();

	// 是否開慢軸
	port.RtsEnable = false;

	// 如果沒有開成功，或是搶 Port 會報錯
	if (!port.IsOpen)
	{
		cout << "OCT 的 COM 打不開!!" << endl;
		return;
	}
	#pragma endregion
	#pragma region 開始掃描
	float LV_65 = 65;
	unsigned int SampleCount = 2048;
	OCT_DataLen = 1;
	OCT_ErrorBoolean = false;
	System::String^ ErrorString = gcnew System::String("");
	System::String^ SaveFileName_C = gcnew System::String(SaveFileName.toStdString().c_str());

	OCT64::OCT64::StartCap(
		OCT_DeviceID,						// 裝置 ID
		OCT_HandleOut,						// Handle (要傳給 Scan 的)
		LV_65,								// 直流
		SampleCount,						// 2048
		OCT_DataLen,						// 資料長度
		NeedSave_RawData,					// 這個好像是要步要 output
		SaveFileName_C,						// 儲存位置
		OCT_ErrorBoolean,					// 是否要有 Error
		ErrorString							// 錯誤訊息
	);
	cout << "OCT StartCap Error String: " << MarshalString(ErrorString) << endl;

	// 要接的 Array
	cli::array<unsigned short>^ OutDataArray = gcnew cli::array<unsigned short>(OCT_PIC_SIZE);				// 暫存的 Array
	unsigned short* Final_OCT_Array = new unsigned short[OCT_PIC_SIZE];										// 取值得 Array
	unsigned short* Temp_OCT_Pointer = Final_OCT_Array;														// 暫存，因為上面那個會一直位移 (別問我，前人就這樣寫= =)
	char* Final_OCT_Char = new char[OCT_PIC_SIZE * 2];														// 最後要丟到 Cuda 的 Array

	// 動慢軸
	OCT_AllDataLen = OCT_DataLen * 2;

	// 掃描
	cli::array<unsigned short>^ inputArray = gcnew cli::array<unsigned short>(OCT_PIC_SIZE);
	OCT64::OCT64::Scan(
		OCT_HandleOut,					// Handle
		OCT_AllDataLen,					// 資料大小
		inputArray,						// 這個沒有用
		OutDataArray,					// 掃描的結果
		ErrorString						// 錯誤訊息
	);
	cout << "Scan Error String: " << MarshalString(ErrorString) << endl;

	// cli Array 轉到 manage array
	pin_ptr<unsigned short> pinPtrArray = &OutDataArray[OutDataArray->GetLowerBound(0)];
	memcpy(Final_OCT_Array, pinPtrArray, sizeof(unsigned short) * OCT_PIC_SIZE);
	#pragma endregion
	#pragma region 關閉已開的 Port
	OCT64::OCT64::AboutADC(OCT_DeviceID);
	port.Close();
	#pragma endregion
	#pragma region 丟到 Cuda 上解資料
	// 要先轉成 Char
	OCT_DataType_Transfrom(Final_OCT_Array, OCT_PIC_SIZE, Final_OCT_Char);

	// 要將資料轉到陣列上
	RawDataProperty prop = DManager.prop;
	cudaV2.SingleRawDataToPointCloud(
		Final_OCT_Char, OCT_PIC_SIZE,
		prop.SizeX, prop.SizeZ,
		prop.ShiftValue, prop.K_Step, prop.CutValue
	);
	// cudaV2.SingleRawDataToPointCloud(buffer.data(), bufferSize, 250, 2048, 37 * 4 - 4, 2, 10);
	#pragma endregion
	#pragma region 刪除 New 出來的 Array
	delete Final_OCT_Array;
	delete Final_OCT_Char;
	#pragma endregion
}
void RawDataManager::Scan_MultiData_FromDeviceV2(QString SaveFileName, bool NeedSave_RawData)
{
	#pragma region 初始化裝置
	OCT64::OCT64::Init(
		4,
		OCT_DeviceID
	);
	#pragma endregion
	#pragma region 開 Port
	SerialPort port(gcnew System::String(OCTDevicePort.c_str()), 9600);
	port.Open();

	// 是否開慢軸
	port.RtsEnable = true;

	// 如果沒有開成功，或是搶 Port 會報錯
	if (!port.IsOpen)
	{
		cout << "OCT 的 COM 打不開!!" << endl;
		return;
	}
	#pragma endregion
	#pragma region 開始掃描
	float LV_65 = 65;
	unsigned int SampleCount = 2048;
	OCT_DataLen = 1;
	OCT_ErrorBoolean = false;
	System::String^ ErrorString = gcnew System::String("");
	System::String^ SaveFileName_C = gcnew System::String(SaveFileName.toStdString().c_str());

	OCT64::OCT64::StartCap(
		OCT_DeviceID,						// 裝置 ID
		OCT_HandleOut,						// Handle (要傳給 Scan 的)
		LV_65,								// ?
		SampleCount,						// 2048
		OCT_DataLen,						// 資料長度
		NeedSave_RawData,							// 這個好像是要步要 output
		SaveFileName_C,						// 儲存位置
		OCT_ErrorBoolean,					// 是否要有 Error
		ErrorString							// 錯誤訊息
	);

	// 要接的 Array
	cli::array<unsigned short>^ OutDataArray = gcnew cli::array<unsigned short>(OCT_PIC_SIZE);				// 暫存的 Array
	unsigned short* Final_OCT_Array = new unsigned short[OCT_PIC_SIZE * 125];								// 取值得 Array
	unsigned short* Temp_OCT_Pointer = Final_OCT_Array;														// 暫存，因為上面那個會一直位移 (別問我，前人就這樣寫= =)
	char* Final_OCT_Char = new char[OCT_PIC_SIZE * 250];													// 最後要丟到 Cuda 的 Array

	// 動慢軸
	OCT_AllDataLen = OCT_DataLen * 2;
	int PicNumber = 0;
	while (PicNumber < 125) {
		// 掃描
		cli::array<unsigned short>^ inputArray = gcnew cli::array<unsigned short>(OCT_PIC_SIZE);
		OCT64::OCT64::Scan(
			OCT_HandleOut,					// Handle
			OCT_AllDataLen,					// 資料大小
			inputArray,						// 這個沒有用
			OutDataArray,					// 掃描的結果
			ErrorString						// 錯誤訊息
		);

		// cli Array 轉到 manage array
		pin_ptr<unsigned short> pinPtrArray = &OutDataArray[OutDataArray->GetLowerBound(0)];
		memcpy(Temp_OCT_Pointer, pinPtrArray, sizeof(unsigned short) * OCT_PIC_SIZE);
		Temp_OCT_Pointer += OCT_PIC_SIZE;


		// 繼續往下掃
		PicNumber++;
	}
	#pragma endregion
	#pragma region 關閉已開的 Port
	OCT64::OCT64::AboutADC(OCT_DeviceID);
	port.RtsEnable = false;
	port.Close();
	#pragma endregion
	#pragma region 丟到 Cuda 上解資料
	// 要先轉成 Char
	OCT_DataType_Transfrom(Final_OCT_Array, OCT_PIC_SIZE * 125, Final_OCT_Char);

	// 要將資料轉到 DManager.rawDP 上
	RawDataProperty prop = DManager.prop;
	cudaV2.SingleRawDataToPointCloud(
		Final_OCT_Char, OCT_PIC_SIZE * 250,
		prop.SizeX, prop.SizeZ,
		prop.ShiftValue, prop.K_Step, prop.CutValue
	);
	// cudaV2.RawDataToPointCloud(buffer.data(), bufferSize, 250, 250, 2048, 37 * 4 - 4, 2, 10);
	#pragma endregion
	#pragma region 刪除 New 出來的 Array
	delete Final_OCT_Array;
	delete Final_OCT_Char;
	#pragma endregion
}

// Helper Function
int RawDataManager::LerpFunction(int lastIndex, int lastValue, int nextIndex, int nextValue, int index)
{
	return (index - lastIndex) * (nextValue - lastValue) / (nextIndex - lastIndex) + lastValue;
}
QImage RawDataManager::Mat2QImage(cv::Mat const& src, int Type)
{
	cv::Mat temp;												// make the same cv::Mat
	if (Type == CV_8UC3)
		cvtColor(src, temp, CV_BGR2RGB);						// cvtColor Makes a copt, that what i need
	else if (Type == CV_32F)
	{
		src.convertTo(temp, CV_8U, 255);
		cvtColor(temp, temp, CV_GRAY2RGB);						// cvtColor Makes a copt, that what i need
	}
	QImage dest((const uchar *)temp.data, temp.cols, temp.rows, temp.step, QImage::Format_RGB888);
	dest.bits();												// enforce deep copy, see documentation 
																// of QImage::QImage ( const uchar * data, int width, int height, Format format )
	return dest;
}
string RawDataManager::MarshalString(System::String^ s)
{
	using namespace System::Runtime::InteropServices;
	const char* chars =
		(const char*)(Marshal::StringToHGlobalAnsi(s)).ToPointer();
	string os = chars;
	Marshal::FreeHGlobal(System::IntPtr((void*)chars));
	return os;
}
void RawDataManager::OCT_DataType_Transfrom(unsigned short *input, int inputlen, char *output)
{
	int outputlen_tmp = 0;
	for (int i = 0; i < inputlen; i++)
	{
		unsigned short tmp_short = input[i];
		tmp_short = tmp_short >> 8;

		output[outputlen_tmp] = input[i];
		outputlen_tmp++;

		output[outputlen_tmp] = tmp_short;
		outputlen_tmp++;
	}
}
vector<GlobalRegistration::Point3D> RawDataManager::ConvertQVector2Point3D(QVector<QVector3D> PointArray)
{
	vector<GlobalRegistration::Point3D> pc;
	for (int i = 0; i < PointArray.size(); i++)
	{
		GlobalRegistration::Point3D point3D;
		QVector3D currentP = PointArray[i];
		point3D.x() = currentP.x();
		point3D.y() = currentP.y();
		point3D.z() = currentP.z();
		pc.push_back(point3D);
	}
	return pc;
}
void RawDataManager::super4PCS_Align(std::vector<GlobalRegistration::Point3D> *PC1, std::vector<GlobalRegistration::Point3D> *PC2, int max_time_seconds)
{
	clock_t t1, t2;
	t1 = clock();

	// Delta (see the paper).
	double delta = 0.1;

	// Estimated overlap (see the paper).
	double overlap = 0.40;

	// Threshold of the computed overlap for termination. 1.0 means don't terminate
	// before the end.
	double thr = 0.35;

	// Maximum norm of RGB values between corresponded points. 1e9 means don't use.
	double max_color = 150;

	// Number of sampled points in both files. The 4PCS allows a very aggressive
	// sampling.
	int n_points = 500;

	// Maximum angle (degrees) between corresponded normals.
	double norm_diff = 20;

	// Maximum allowed computation time.
	//max_time_seconds = 1;//500

	bool use_super4pcs = true;

	// maximum per-dimension angle, check return value to detect invalid cases
	double max_angle = 20;

	//==========//

	//vector<Point3D> set1, set2;
	vector<cv::Point2f> tex_coords1, tex_coords2;
	vector<cv::Point3f> normals1, normals2;
	vector<tripple> tris1, tris2;
	vector<std::string> mtls1, mtls2;

	IOManager iomananger;


	// super4pcs matcher
	GlobalRegistration::Match4PCSOptions options;


	//cv::Mat mat_rot = cv::Mat::eye(3, 3, CV_64F);
	GlobalRegistration::Match4PCSBase::MatrixType *mat;
	mat = new GlobalRegistration::Match4PCSBase::MatrixType;


	bool overlapOk = options.configureOverlap(overlap, 0.35f);
	if (!overlapOk) {
		std::cerr << "Invalid overlap configuration. ABORT" << std::endl;
		/// TODO Add proper error codes

	}

	//overlap = options.getOverlapEstimation();
	options.sample_size = n_points;
	options.max_normal_difference = norm_diff;
	options.max_color_distance = max_color;
	options.max_time_seconds = max_time_seconds;
	options.delta = delta;
	options.max_angle = max_angle;

	GlobalRegistration::Match4PCSOptions::Scalar Estimation;

	Estimation = options.getOverlapEstimation();
	std::cout << "getOverlapEstimation:" << Estimation << std::endl;
	//// Match and return the score (estimated overlap or the LCP).  
	typename GlobalRegistration::Point3D::Scalar score = 0;
	int v;

	constexpr GlobalRegistration::Utils::LogLevel loglvl = GlobalRegistration::Utils::Verbose;
	//cv::Mat mat_rot(*mat);

	//template <typename Visitor>
	using TrVisitorType = typename std::conditional <loglvl == GlobalRegistration::Utils::NoLog,
		GlobalRegistration::Match4PCSBase::DummyTransformVisitor,
		TransformVisitor>::type;

	GlobalRegistration::Utils::Logger logger(loglvl);

	QMatrix4x4 matrix;
	matrix.setToIdentity();

	try {

		if (use_super4pcs) {
			GlobalRegistration::MatchSuper4PCS *matcher;
			matcher = new GlobalRegistration::MatchSuper4PCS(options, logger);

			cout << "Use Super4PCS" << endl;
			score = matcher->ComputeTransformation(*PC1, PC2, *mat);

			//cout << "Mat Mat" << endl;
			//cout << (*mat) << endl;

			//for (int i = 0; i < 4; i++)
			//	for (int j = 0; j < 4; j++)
			//		matrix(i, j) = (*mat)(i, j);
			//qDebug() << matrix << endl;

			cout << std::endl;
			cout << "score_in:" << score << std::endl;
		}
		else {
			GlobalRegistration::Match4PCS *matcher;
			matcher = new GlobalRegistration::Match4PCS(options, logger);
			std::cout << "Use old 4PCS" << std::endl;
			score = matcher->ComputeTransformation(*PC1, PC2, *mat);
			std::cout << "score_in:" << score << std::endl;
		}
	}
	catch (const std::exception& e) {
		cout << "[Error]: " << e.what() << '\n';
		cout << "Aborting with code -2 ..." << endl;
		return;
	}
	catch (...) {
		std::cout << "[Unknown Error]: Aborting with code -3 ..." << std::endl;
		return;
	}
	t2 = clock();
	//qDebug() << "Match done t: " << (t2 - t1) / (double)(CLOCKS_PER_SEC) << " s  ";
	//final_score = score;
	cout << "Score: " << score << endl;
	//cerr << score << endl;
}