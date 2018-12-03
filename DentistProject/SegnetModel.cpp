#include "SegNetModel.h"

SegNetModel::SegNetModel()
{
	google::InitGoogleLogging("DentistProject.exe");
	Caffe::set_mode(Caffe::GPU);
	LabelImg = imread(LUT_file, 1);
}
SegNetModel::~SegNetModel()
{
	if (SegNet != NULL)
		delete SegNet;
}

void SegNetModel::Load(string ModelDefPath, string ModelPath)
{
	// TEST NET
	SegNet = new Net<float>(ModelDefPath, TEST);
	SegNet->CopyTrainedLayersFrom(ModelPath);
	this->ModelDefPath = ModelDefPath;
	this->ModelPath = ModelPath;

	//////////////////////////////////////////////////////////////////////////
	// 錯誤判斷是否有輸入輸出
	//////////////////////////////////////////////////////////////////////////
	CHECK_EQ(SegNet->num_inputs(), 1) << "Network should have exactly one input.";
	CHECK_EQ(SegNet->num_outputs(), 1) << "Network should have exactly one output.";

	// Input 變數
	Blob<float>* input_layer = SegNet->input_blobs()[0];
	BatchSize = input_layer->shape()[0];
	InputSize = NetworkSize(input_layer->width(), input_layer->height(), input_layer->channels());
	cout << "========== Input ==========" << endl;
	cout << "BatchSize\t=> " << BatchSize << endl;
	cout << "Channel\t\t=> " << InputSize.NumChannels << endl;
	cout << "Width\t\t=> " << InputSize.Width << endl;
	cout << "Height\t\t=> " << InputSize.Height << endl;

	// Output 變數
	Blob<float>* output_layer = SegNet->output_blobs()[0];
	OutputSize = NetworkSize(output_layer->width(), output_layer->height(), output_layer->channels());
	cout << "========== Output ==========" << endl;
	cout << "Channel\t\t=> " << OutputSize.NumChannels << endl;
	cout << "Width\t\t=> " << OutputSize.Width << endl;
	cout << "Height\t\t=> " << OutputSize.Height << endl;
	cout << "========== End ==========" << endl;
}
void SegNetModel::ReshapeToMultiBatch(int NewBatchSize)
{
	// 重開更改 Batch 大小
	SegNet->input_blobs()[0]->Reshape(NewBatchSize, InputSize.NumChannels, InputSize.Height, InputSize.Width);
	SegNet->output_blobs()[0]->Reshape(NewBatchSize, OutputSize.NumChannels, OutputSize.Height, OutputSize.Width);
	SegNet->Reshape();

	// 設定新的 Batch Size
	BatchSize = SegNet->input_blobs()[0]->shape()[0];
	cout << "設定 Net Batch Size: " << BatchSize << endl;
}
Mat SegNetModel::Predict(Mat &img)
{
	#pragma region 預先判斷是否用錯
	// 這邊一定要是只使用單通道
	assert(BatchSize == 1);
	#pragma endregion
	#pragma region 前處理
	WrapSingleInputLayer();			// 要重新產生一塊 Memory 去
	SinglePreprocess(img);
	#pragma endregion
	#pragma region 跑網路結果
	// 測試時間
	chrono::steady_clock::time_point BeginTime = chrono::steady_clock::now();

	SegNet->Forward();

	// Output時間
	chrono::steady_clock::time_point EndTime = chrono::steady_clock::now();
	cout << "處理時間(Multi) = " << (chrono::duration_cast<chrono::microseconds>(EndTime - BeginTime).count()) / 1000000.0 << " sec" << endl;
	#pragma endregion
	#pragma region 轉成 Output
	Blob<float>* output_layer = SegNet->output_blobs()[0];

	// 拿出所有的資料來
	const float* DataBegin = output_layer->cpu_data();
	const float* DataEnd = DataBegin + OutputSize.NumChannels * OutputSize.Width * OutputSize.Height;
	vector<float> OutputValue = vector<float>(DataBegin, DataEnd);

	// 將值塞進去
	Mat PredictImage(OutputSize.cvSize(), CV_8U);
	#pragma omp parallel for num_threads(4)
	for (int rowIndex = 0; rowIndex < OutputSize.Height; rowIndex++)
		for (int colIndex = 0; colIndex < OutputSize.Width; colIndex++)
		{
			vector<float> ProbValue;
			for (int c = 0; c < OutputSize.NumChannels ; c++)
				ProbValue.push_back(OutputValue[c * OutputSize.Height * OutputSize.Width + rowIndex* OutputSize.Width + colIndex]);
			PredictImage.at<uchar>(rowIndex, colIndex) = ArgMax(ProbValue);
		}
	return PredictImage;
	#pragma endregion
}
vector<Mat> SegNetModel::Predict(vector<Mat> InputArray)
{
	#pragma region 預先判斷是否用錯
	// 判斷現在是否是 Multi Input 的
	assert(BatchSize > 1);
	#pragma endregion
	#pragma region 多個 For 迴圈
	// 測試時間
	chrono::steady_clock::time_point BeginTime = chrono::steady_clock::now();

	int iterTimes = ceil((float)InputArray.size() / BatchSize);						// 總共要跑幾個 iter
	int LastIterCount = InputArray.size() - BatchSize * (iterTimes - 1);			// 最後一個 iter 有幾張

	// Final 結果
	vector<Mat> PredictImageArray;

	// 創建要 Mapping 的 vector
	for (int i = 0; i < iterTimes; i++)
	{
		#pragma region 先拿某一部分
		vector<Mat> vectorArray;
		for (int j = 0; j < BatchSize; j++)
		{
			int index = i * BatchSize + j;
			if (InputArray.size() > index)
				vectorArray.push_back(InputArray[index]);
		}
		#pragma endregion
		#pragma region 前處理
		WrapMultiInputLayer();
		MultiPreprocess(vectorArray);
		#pragma endregion
		#pragma region 跑網路結果
		SegNet->Forward();
		#pragma endregion
		#pragma region 轉成 Output
		Blob<float>* output_layer = SegNet->output_blobs()[0];

		// 拿出所有的資料來
		const float* DataBegin = output_layer->cpu_data();
		const float* DataEnd = DataBegin + BatchSize * OutputSize.NumChannels * OutputSize.Width * OutputSize.Height;
		vector<float> OutputValue(DataBegin, DataEnd);

		// 這邊要多加上一個條件
		// 如果是在最後一個的話 (iterTimes)
		// 那代表會有可能有 Empty
		for (int batchIndex = 0; batchIndex < BatchSize && (!(i == iterTimes - 1) || batchIndex < LastIterCount); batchIndex++)
		{
			Mat PredictImage(OutputSize.cvSize(), CV_8U);
			#pragma omp parallel for num_threads(4)
			for (int rowIndex = 0; rowIndex < OutputSize.Height; rowIndex++)
				for (int colIndex = 0; colIndex < OutputSize.Width; colIndex++)
				{
					vector<float> ProbValue;
					for (int c = 0; c < OutputSize.NumChannels; c++)
						ProbValue.push_back(OutputValue[
							batchIndex * OutputSize.Height * OutputSize.Width * OutputSize.NumChannels +			// 每個 Batch
							c * OutputSize.Height * OutputSize.Width +												// 每個結果 (1 ~ numChnagel)
							rowIndex* OutputSize.Width +															// 每個 Row
							colIndex																				// 在第幾個 Col
						]);
					PredictImage.at<uchar>(rowIndex, colIndex) = ArgMax(ProbValue);
				}
			PredictImageArray.push_back(PredictImage);
		}
		#pragma endregion
	}

	// 要確定 Input && Output 的 Size 是一樣的
	assert(PredictImageArray.size() == InputArray.size());

	// Reload Net (好像他沒有 Free 的 Function)
	delete SegNet;
	SegNet = new Net<float>(ModelDefPath, TEST);
	SegNet->CopyTrainedLayersFrom(ModelPath);
	ReshapeToMultiBatch(BatchSize);

	// Output時間
	chrono::steady_clock::time_point EndTime = chrono::steady_clock::now();
	cout << "處理時間(SegNet) = " << (chrono::duration_cast<chrono::microseconds>(EndTime - BeginTime).count()) / 1000000.0 << " sec" << endl;
	return PredictImageArray;
	#pragma endregion
}
Mat SegNetModel::Visualization(Mat img)
{
	// 從灰階轉成 RGB
	cvtColor(img.clone(), img, CV_GRAY2BGR);

	Mat output_image;

	// 對應回 LUT 的資料
	LUT(img, LabelImg, output_image);
	return output_image;
}

//////////////////////////////////////////////////////////////////////////
// Helper Function
//////////////////////////////////////////////////////////////////////////
void SegNetModel::WrapSingleInputLayer()
{
	// 這邊一定要是只使用單通道
	assert(BatchSize == 1);

	Blob<float>* input_layer = SegNet->input_blobs()[0];

	int width = InputSize.Width;
	int height = InputSize.Height;
	float* input_data = input_layer->mutable_cpu_data();

	// 位置修正
	InputChannelPointer.clear();
	for (int i = 0; i < InputSize.NumChannels; ++i) 
	{
		Mat channel(height, width, CV_32FC1, input_data);
		InputChannelPointer.push_back(channel);
		input_data += width * height;
	}
}
void SegNetModel::WrapMultiInputLayer()
{
	// 這邊一定要是只使用單通道
	assert(BatchSize > 1);

	//SegNet->Init()
	Blob<float>* input_layer = SegNet->input_blobs()[0];

	int width = InputSize.Width;
	int height = InputSize.Height;
	float* input_data = input_layer->mutable_cpu_data();

	// 位置修正
	InputChannelPointer.clear();
	for(int b = 0; b < BatchSize; b ++)
		for (int i = 0; i < InputSize.NumChannels; ++i)
		{
			Mat channel(height, width, CV_32FC1, input_data);
			InputChannelPointer.push_back(channel);
			input_data += width * height;
		}
}
void SegNetModel::SinglePreprocess(Mat& img) 
{
	Mat sample;
	if (img.channels() == 3 && InputSize.NumChannels == 1)
		cvtColor(img, sample, COLOR_BGR2GRAY);
	else if (img.channels() == 4 && InputSize.NumChannels == 1)
		cvtColor(img, sample, COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && InputSize.NumChannels == 3)
		cvtColor(img, sample, COLOR_BGRA2BGR);
	else if (img.channels() == 1 && InputSize.NumChannels == 3)
		cvtColor(img, sample, COLOR_GRAY2BGR);
	else
		sample = img;
	//DebugMat(sample, Type8UC3);

	Mat sample_resized;
	if (sample.size() != InputSize.cvSize())
		resize(sample, sample_resized, InputSize.cvSize());
	else
		sample_resized = sample;

	Mat sample_float;
	if (InputSize.NumChannels == 3)
		sample_resized.convertTo(sample_float, CV_32FC3);
	else
		sample_resized.convertTo(sample_float, CV_32FC1);

	//DebugMat(sample_float, Type32FC3);
	split(sample_float, InputChannelPointer);


	// Check 是否可以凹回來
	CHECK(reinterpret_cast<float*>(InputChannelPointer.at(0).data) 
		== SegNet->input_blobs()[0]->cpu_data())
		<< "Input channels are not wrapping the input layer of the network.";
}
void SegNetModel::MultiPreprocess(vector<Mat> imgArray)
{
	for (int i = 0; i < imgArray.size(); i++)
	{
		Mat sample;
		Mat img = imgArray[i];
		if (img.channels() == 3 && InputSize.NumChannels == 1)
			cvtColor(img, sample, COLOR_BGR2GRAY);
		else if (img.channels() == 4 && InputSize.NumChannels == 1)
			cvtColor(img, sample, COLOR_BGRA2GRAY);
		else if (img.channels() == 4 && InputSize.NumChannels == 3)
			cvtColor(img, sample, COLOR_BGRA2BGR);
		else if (img.channels() == 1 && InputSize.NumChannels == 3)
			cvtColor(img, sample, COLOR_GRAY2BGR);
		else
			sample = img;
		//DebugMat(sample, Type8UC3);

		Mat sample_resized;
		if (sample.size() != InputSize.cvSize())
			resize(sample, sample_resized, InputSize.cvSize());
		else
			sample_resized = sample;

		Mat sample_float;
		if (InputSize.NumChannels == 3)
			sample_resized.convertTo(sample_float, CV_32FC3);
		else
			sample_resized.convertTo(sample_float, CV_32FC1);

		//DebugMat(sample_float, Type32FC3);
		// Copy 過去
		vector<Mat> SplitPointer;
		split(sample_float, SplitPointer);
		for (int c = 0; c < InputSize.NumChannels; c++)
		{
			SplitPointer[c].copyTo(InputChannelPointer[i * InputSize.NumChannels + c]);
			//InputChannelPointer[i * InputSize.NumChannels + c] = SplitPointer[c];
		}
	}

	// Check 是否可以凹回來
	CHECK(reinterpret_cast<float*>(InputChannelPointer.at(0).data)
		== SegNet->input_blobs()[0]->cpu_data())
		<< "Input channels are not wrapping the input layer of the network.";
}

int SegNetModel::ArgMax(vector<float> ValueArray)
{
	int TempIndex = 0;
	float TempValue = ValueArray[0];
	for (int i= 1; i < ValueArray.size(); i++)
		if (TempValue < ValueArray[i])
		{
			TempValue = ValueArray[i];
			TempIndex = i;
		}
	return TempIndex;
}

// Debug 用的
void SegNetModel::DebugMat(Mat img, DebugMatType t)
{
	switch (t)
	{
	case Type8U:
	{
		for (int r = 0; r < img.rows; r++)
		{
			for (int c = 0; c < img.cols; c++)
			{
				uchar color = img.at<uchar>(r, c);
				cout << int(color) << " ";
			}
			cout << endl;
			break;
		}
	}
	break;
	case Type8UC3:
	{
		for (int r = 0; r < img.rows; r++)
		{
			for (int c = 0; c < img.cols; c++)
			{
				Vec3b color = img.at<Vec3b>(r, c);
				cout << (int)(color[0]) << "," << int(color[1]) << "," << int(color[2]) << " ";
			}
			cout << endl;
			break;
		}
	}
	break;
	case Type32F:
	{
		for (int r = 0; r < img.rows; r++)
		{
			for (int c = 0; c < img.cols; c++)
			{
				float color = img.at<float>(r, c);
				cout << color << " ";
			}
			cout << endl;
			break;
		}
	}
	case Type32FC3:
	{
		for (int r = 0; r < img.rows; r++)
		{
			for (int c = 0; c < img.cols; c++)
			{
				Vec3f color = img.at<Vec3f>(r, c);
				cout << color[0] << "," << color[1] << "," << color[2] << " ";
			}
			cout << endl;
			break;
		}
	}
	break;
	default:
		break;
	}
}