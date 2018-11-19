#include "SegnetModel.h"

SegnetModel::SegnetModel()
{
	google::InitGoogleLogging("DentistDemo.exe");
	Caffe::set_mode(Caffe::GPU);
	LabelImg = imread(LUT_file, 1);
}
SegnetModel::~SegnetModel()
{
	if (SegNet != NULL)
		delete SegNet;
}

void SegnetModel::Load(string ModelDefPath, string ModelPath)
{
	// TEST NET
	SegNet = new Net<float>(ModelDefPath, TEST);
	SegNet->CopyTrainedLayersFrom(ModelPath);

	//////////////////////////////////////////////////////////////////////////
	// 錯誤判斷是否有輸入輸出
	//////////////////////////////////////////////////////////////////////////
	CHECK_EQ(SegNet->num_inputs(), 1) << "Network should have exactly one input.";
	CHECK_EQ(SegNet->num_outputs(), 1) << "Network should have exactly one output.";

	// Input 變數
	Blob<float>* input_layer = SegNet->input_blobs()[0];
	InputSize = NetworkSize(input_layer->width(), input_layer->height(), input_layer->channels());
	cout << "========== Input ==========" << endl;
	cout << "Channel\t=> " << InputSize.NumChannels << endl;
	cout << "Width\t=> " << InputSize.Width << endl;
	cout << "Height\t=> " << InputSize.Height << endl;

	// Output 變數
	Blob<float>* output_layer = SegNet->output_blobs()[0];
	OutputSize = NetworkSize(output_layer->width(), output_layer->height(), output_layer->channels());
	cout << "========== Output ==========" << endl;
	cout << "Channel\t=> " << OutputSize.NumChannels << endl;
	cout << "Width\t=> " << OutputSize.Width << endl;
	cout << "Height\t=> " << OutputSize.Height << endl;
	cout << "========== End ==========" << endl;
}
Mat SegnetModel::Predict(Mat &img)
{
	#pragma region 前處理
	WrapInputLayer();			// 要重新產生一塊 Memory 去
	Preprocess(img);
	#pragma endregion
	#pragma region 跑網路結果
	// 測試時間
	//chrono::steady_clock::time_point BeginTime = chrono::steady_clock::now();

	SegNet->Forward();

	// Output時間
	//chrono::steady_clock::time_point EndTime = chrono::steady_clock::now();
	//cout << "處理時間 = " << (chrono::duration_cast<chrono::microseconds>(EndTime - BeginTime).count()) / 1000000.0 << " sec" << endl;
	#pragma endregion
	#pragma region 轉成 Output
	Blob<float>* output_layer = SegNet->output_blobs()[0];

	// 拿出所有的資料來
	const float* DataBegin = output_layer->cpu_data();
	const float* DataEnd = DataBegin + OutputSize.NumChannels * OutputSize.Width * OutputSize.Height;
	vector<float> OutputValue = vector<float>(DataBegin, DataEnd);

	// 將值塞進去
	Mat PredictImage(OutputSize.cvSize(), CV_8U);
	#pragma omp parallel for
	for (int rowIndex = 0; rowIndex < OutputSize.Height; rowIndex++)
		for (int colIndex = 0; colIndex < OutputSize.Width; colIndex++)
		{
			vector<float> ProbValue;
			for (int i = 0; i < 5; i++)
				ProbValue.push_back(OutputValue[i * OutputSize.Height * OutputSize.Width + rowIndex* OutputSize.Width + colIndex]);
			PredictImage.at<uchar>(rowIndex, colIndex) = ArgMax(ProbValue);
		}
	return PredictImage;
	#pragma endregion
}
Mat SegnetModel::Visualization(Mat img)
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
void SegnetModel::WrapInputLayer()
{
	Blob<float>* input_layer = SegNet->input_blobs()[0];

	int width = InputSize.Width;
	int height = InputSize.Height;
	float* input_data = input_layer->mutable_cpu_data();

	// 位置修正
	InputChannelPointer.clear();
	for (int i = 0; i < InputSize.NumChannels; ++i) 
	{
		cv::Mat channel(height, width, CV_32FC1, input_data);
		InputChannelPointer.push_back(channel);
		input_data += width * height;
	}
}
void SegnetModel::Preprocess(Mat& img) {
	Mat sample;
	if (img.channels() == 3 && InputSize.NumChannels == 1)
		cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	else if (img.channels() == 4 && InputSize.NumChannels == 1)
		cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && InputSize.NumChannels == 3)
		cvtColor(img, sample, cv::COLOR_BGRA2BGR);
	else if (img.channels() == 1 && InputSize.NumChannels == 3)
		cvtColor(img, sample, cv::COLOR_GRAY2BGR);
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
int SegnetModel::ArgMax(vector<float> ValueArray)
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
void SegnetModel::DebugMat(Mat img, DebugMatType t)
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
