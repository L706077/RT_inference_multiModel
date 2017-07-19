#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <sys/stat.h>
#include <cmath>
#include <time.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <string>
#include <utility>
#include <stdlib.h>

/////////////get class path head////////////
#include <unistd.h>
#include <dirent.h>
//////////////////////////////////////

#include "NvInfer.h"
#include "NvCaffeParser.h"

//using namespace nvinfer1;
//using namespace nvcaffeparser1;
//using namespace cv;
typedef std::pair<std::string,float> mate;
#define CHECK(status)									\
{														\
	if (status != 0)									\
	{													\
		std::cout << "Cuda failure: " << status;		\
		abort();										\
	}													\
}


// stuff we know about the network and the caffe input/output blobs
//static const int INPUT_H = 227;
//static const int INPUT_W = 227;
//static const int CHANNEL_NUM = 3;
//static const int OUTPUT_SIZE = 1000;
static const int INPUT_H = 192;
static const int INPUT_W = 192;
static const int CHANNEL_NUM = 3;
//static const int OUTPUT_SIZE = 1498; //1498
int OUTPUT_SIZE1 = 1498; //  ********************Define by yourself*****************   
int OUTPUT_SIZE2 = 1694; //  ********************Define by yourself***************** 


//===================================================
const std::string Path_   = "./fr_model/";
//===================================================
const std::string Model_  = "fr_1498.caffemodel";
const std::string Deploy_ = "deploy.prototxt";
const std::string Image_  = "1.jpg";
const std::string Mean_   = "mean.binaryproto";
const std::string Label_  = "labels.txt";
//===================================================
const std::string Model2_  = "caffe_Face_VGG16_1694.caffemodel";
const std::string Deploy2_ = "deploy2.prototxt";
const std::string Image2_  = "2.jpg";
const std::string Mean2_   = "mean2.binaryproto";
const std::string Label2_  = "labels2.txt";
//===================================================
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "softmax";    //  ********************Define by yourself***************** 
const char* OUTPUT_BLOB_NAME2 = "softmax";   //  ********************Define by yourself***************** 
//const char* OUTPUT_BLOB_NAME = "fc11_dropout";
//===================================================

//===================================================
// Logger for GIE info/warning/errors
class Logger : public nvinfer1::ILogger			
{
	void log(Severity severity, const char* msg) override
	{
		// suppress info-level messages
		if (severity != Severity::kINFO)
			std::cout << msg << std::endl;
	}
} gLogger1;

class Logger2 : public nvinfer1::ILogger			
{
	void log(Severity severity, const char* msg) override
	{
		// suppress info-level messages
		if (severity != Severity::kINFO)
			std::cout << msg << std::endl;
	}
} gLogger2;

std::string locateFile(const std::string& input)
{
	std::string file = Path_ + input;
	struct stat info;
	int i, MAX_DEPTH = 1;
	for (i = 0; i < MAX_DEPTH && stat(file.c_str(), &info); i++)
		file = "../" + file;

	return file;
}


void caffeToGIEModel2(const std::string& deployFile1,				// name for caffe prototxt1
		      const std::string& deployFile2,				// name for caffe prototxt2
		      const std::string& modelFile1,				// name for model1 
                      const std::string& modelFile2,				// name for model2 
					 const std::vector<std::string>& outputs1,   // network outputs1
					 const std::vector<std::string>& outputs2,   // network outputs2
					 unsigned int maxBatchSize,					// batch size - NB must be at least as large as the batch we want to run with)
					 nvinfer1::IHostMemory *&gieModelStream1,
					 nvinfer1::IHostMemory *&gieModelStream2
)    // output buffer for the GIE model
{

	nvinfer1::IBuilder* builder1 = nvinfer1::createInferBuilder(gLogger1);
	nvinfer1::IBuilder* builder2 = nvinfer1::createInferBuilder(gLogger2);
											std::cout<<"TEST1:"<<std::endl;
	nvinfer1::INetworkDefinition* network1 = builder1->createNetwork();
	nvinfer1::INetworkDefinition* network2 = builder2->createNetwork();
											std::cout<<"TEST2:"<<std::endl;
	nvcaffeparser1::ICaffeParser* parser1 = nvcaffeparser1::createCaffeParser();
	nvcaffeparser1::ICaffeParser* parser2 = nvcaffeparser1::createCaffeParser();
											std::cout<<"TEST3:"<<std::endl;
	const nvcaffeparser1::IBlobNameToTensor* blobNameToTensor1;
	const nvcaffeparser1::IBlobNameToTensor* blobNameToTensor2;
											std::cout<<"TEST4:"<<std::endl;
	blobNameToTensor1 = parser1->parse(locateFile(deployFile1).c_str(), locateFile(modelFile1).c_str(), *network1, nvinfer1::DataType::kFLOAT);

	blobNameToTensor2 = parser2->parse(locateFile(deployFile2).c_str(), locateFile(modelFile2).c_str(), *network2, nvinfer1::DataType::kFLOAT);

											std::cout<<"TEST5:"<<std::endl;
	// specify which tensors are outputs
	for (auto& s : outputs1)
		network1->markOutput(*blobNameToTensor1->find(s.c_str()));

	for (auto& s2 : outputs2)
		network2->markOutput(*blobNameToTensor2->find(s2.c_str()));

											std::cout<<"TEST6:"<<std::endl;
	// Build the engine
	builder1->setMaxBatchSize(maxBatchSize);
	builder1->setMaxWorkspaceSize(1 << 20);

	builder2->setMaxBatchSize(maxBatchSize);
	builder2->setMaxWorkspaceSize(1 << 20);


	//std::cout<<"builder->getMaxWorkspaceSize()"<<std::endl;
	std::cout<<"builder1->getMaxWorkspaceSize()"<< builder1->getMaxWorkspaceSize()<< std::endl;
	std::cout<<"builder2->getMaxWorkspaceSize()"<< builder2->getMaxWorkspaceSize()<< std::endl;
											std::cout<<"TEST7:"<<std::endl;
	//builder->setHalf2Mode(true);
	nvinfer1::ICudaEngine* engine1 = builder1->buildCudaEngine(*network1);
	assert(engine1);

	nvinfer1::ICudaEngine* engine2 = builder2->buildCudaEngine(*network2);
	assert(engine2);
											std::cout<<"TEST8:"<<std::endl;
	// we don't need the network any more, and we can destroy the parser
	network1->destroy();
	network2->destroy();
	
	parser1->destroy();
	parser2->destroy();
											std::cout<<"TEST9:"<<std::endl;
	gieModelStream1 = engine1->serialize();
	gieModelStream2 = engine2->serialize();

	engine1->destroy();
	engine2->destroy();
	
	builder1->destroy();
	builder2->destroy();

	nvcaffeparser1::shutdownProtobufLibrary();
											std::cout<<"TEST10:"<<std::endl;
}


void doInference2(nvinfer1::IExecutionContext& context, float* input, float* output, int batchSize, int output_size_)
{
	const nvinfer1::ICudaEngine& engine = context.getEngine();

	assert(engine.getNbBindings() == 2);
	void* buffers[2];

	int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME), 
		outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

	// create GPU buffers and a stream
	CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_H * INPUT_W * CHANNEL_NUM * sizeof(float)));
	CHECK(cudaMalloc(&buffers[outputIndex], batchSize * output_size_ * sizeof(float)));

	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));

	// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
	CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_H * INPUT_W * CHANNEL_NUM * sizeof(float), cudaMemcpyHostToDevice, stream));

	context.enqueue(batchSize, buffers, stream, nullptr);

        CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * output_size_*sizeof(float), cudaMemcpyDeviceToHost, stream));

	cudaStreamSynchronize(stream);

	// release the stream and the buffers
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputIndex]));
}




static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  	return lhs.first > rhs.first;
}


/* Return the indices of the top N values of p. */
static std::vector<int> Argmax2(const float *p, int N, int output_size_) 
{
  	std::vector<std::pair<float, int> > pairs;
  	for (size_t i = 0; i < output_size_; ++i)
    		pairs.push_back(std::make_pair(p[i], i));
  	std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  	std::vector<int> result;
  	for (int i = 0; i < N; ++i)
    		result.push_back(pairs[i].second);
  	return result;
}


void preprocess(cv::Mat& img, cv::Size input_geometry_)
{
	cv::Mat sample, sample_resized;
	input_geometry_ = cv::Size(INPUT_W, INPUT_H);

	if (img.channels() == 3 && CHANNEL_NUM == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	else if (img.channels() == 4 && CHANNEL_NUM == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && CHANNEL_NUM == 3)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
	else if (img.channels() == 1 && CHANNEL_NUM == 3)
		cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
	else
		sample = img;

	if (sample.size() != input_geometry_)
	    cv::resize(sample, sample_resized, input_geometry_);

	else
	    sample_resized = sample;

	img = sample_resized;
}


void showInferenceResult(const std::string Path, const std::string Label, float *prob_, int outputsize)
{
	std::ifstream labels1(Path + Label);
	std::string line;
	std::vector<std::string> labels1_;
	while (std::getline(labels1, line))
		labels1_.push_back(std::string(line));	
	std::vector<int>maxN = Argmax2(prob_, 5, outputsize);  // find top 5 sort
	std::vector<mate> predictions;        //typedef std::pair<std::string,float> mate;
	for(int i=0;i<5;i++)
	{
		int idx=maxN[i];
		predictions.push_back(std::make_pair(labels1_[idx],prob_[idx]));
	}
	
	// Print the top N predictions. 
	for (size_t i = 0; i < predictions.size(); ++i) {
		mate p = predictions[i];
		std::cout << std::fixed << p.second << " - \""
		<< p.first << "\"" << std::endl;
	}
	std::cout<<" ........ " << std::endl;
	
}


void imageCalculation(cv::Mat img_input, const int INPUT_W, const int INPUT_H, const int CHANNEL_NUM, const float *meanData_, float* data)
{
		cv::Mat Img;
		cv::Size input_geometry_;
		input_geometry_ = cv::Size(INPUT_W, INPUT_H);
		preprocess(img_input,input_geometry_);
		Img = img_input;
		cv::Mat channel[CHANNEL_NUM];
		cv::split(Img,channel);	
		unsigned int fileData[INPUT_H*INPUT_W*CHANNEL_NUM];
		int num_time=0; 
		for(int k=0;k<CHANNEL_NUM;k++)
		{	
			for(int i=0;i<INPUT_H;i++)
			{
				for(int j=0;j<INPUT_W;j++)
				{
					fileData[num_time]=(int)channel[k].at<uchar>(i,j);
					num_time++;			
				}
			}
		}
		for (int i = 0; i < INPUT_H*INPUT_W*CHANNEL_NUM; i++)
		{	
			data[i] = float(fileData[i])-meanData_[i];							
		}		
}



int main(int argc, char** argv)
{
	clock_t t1, t2, t3, t4, t5, t6, t7;

t1=clock();
	// create a GIE model from the caffe model and serialize it to a stream
    	nvinfer1::IHostMemory *gieModelStream1{nullptr};
	nvinfer1::IHostMemory *gieModelStream2{nullptr};
	//caffeToGIEModel(Deploy_, Model_, std::vector < std::string > { OUTPUT_BLOB_NAME }, 1, gieModelStream);
	caffeToGIEModel2(Deploy_, Deploy2_, Model_, Model2_, std::vector < std::string > { OUTPUT_BLOB_NAME }, std::vector < std::string > { OUTPUT_BLOB_NAME2 }, 1, gieModelStream1, gieModelStream2);

t2=clock();
	// deserialize the engine 
	nvinfer1::IRuntime* runtime1 = nvinfer1::createInferRuntime(gLogger1);
	nvinfer1::IRuntime* runtime2 = nvinfer1::createInferRuntime(gLogger2);

	nvinfer1::ICudaEngine* engine1 = runtime1->deserializeCudaEngine(gieModelStream1->data(), gieModelStream1->size(), nullptr);
	nvinfer1::ICudaEngine* engine2 = runtime2->deserializeCudaEngine(gieModelStream2->data(), gieModelStream2->size(), nullptr);

	if (gieModelStream1) gieModelStream1->destroy();
	if (gieModelStream2) gieModelStream2->destroy();

	nvinfer1::IExecutionContext *context1 = engine1->createExecutionContext();
	nvinfer1::IExecutionContext *context2 = engine2->createExecutionContext();
	std::cout<<"engine builded!!!!"<< std::endl;

t3=clock();	
	// parse the mean1 file 
	nvcaffeparser1::ICaffeParser* parser1 = nvcaffeparser1::createCaffeParser();
	nvcaffeparser1::IBinaryProtoBlob* meanBlob1 = parser1->parseBinaryProto(locateFile(Mean_).c_str()); 
	parser1->destroy();
	const float *meanData1 = reinterpret_cast<const float*>(meanBlob1->getData());

	// parse the mean2 file 
	nvcaffeparser1::ICaffeParser* parser2 = nvcaffeparser1::createCaffeParser();
	nvcaffeparser1::IBinaryProtoBlob* meanBlob2 = parser2->parseBinaryProto(locateFile(Mean2_).c_str()); 
	parser2->destroy();
	const float *meanData2 = reinterpret_cast<const float*>(meanBlob2->getData());

t4=clock();
//==================================================================

	float *prob1=(float*)malloc(OUTPUT_SIZE1*sizeof(float));
	cv::Mat img_input = cv::imread(Path_ + Image_ , 1);
	if (!img_input.empty())
	{		
		float data1[INPUT_H*INPUT_W*CHANNEL_NUM];
		imageCalculation(img_input, INPUT_W,INPUT_H, CHANNEL_NUM, meanData1, data1);
		doInference2(*context1, data1, prob1, 1, OUTPUT_SIZE1);    
	}

t5=clock();

	float *prob2=(float*)malloc(OUTPUT_SIZE2*sizeof(float));
	img_input = cv::imread(Path_ + Image2_ , 1);
	if (!img_input.empty())
	{				
		float data2[INPUT_H*INPUT_W*CHANNEL_NUM];
		imageCalculation(img_input, INPUT_W, INPUT_H, CHANNEL_NUM, meanData2, data2);
		doInference2(*context2, data2, prob2, 1, OUTPUT_SIZE2);   
	}

//==================================================================

t6=clock();

	// destroy the everything
	meanBlob1->destroy();
	meanBlob2->destroy();	

	context1->destroy();
	context2->destroy();

	engine1->destroy();
	engine2->destroy();
	
	runtime1->destroy();
	runtime2->destroy();

	showInferenceResult(Path_,Label_,prob1,OUTPUT_SIZE1);
	showInferenceResult(Path_,Label2_,prob2,OUTPUT_SIZE2);
	
	free(prob1);
	free(prob2);

t7=clock();
	
	std::cout<<"t2-t1 time:"<<(double)(t2-t1)/(CLOCKS_PER_SEC)<<"s"<<" (create GIE model) "<<std::endl;
	std::cout<<"t3-t2 time:"<<(double)(t3-t2)/(CLOCKS_PER_SEC)<<"s"<<" (build engine "<<std::endl;
	std::cout<<"t4-t3 time:"<<(double)(t4-t3)/(CLOCKS_PER_SEC)<<"s"<<" (parse mean file) "<<std::endl;
	std::cout<<"t5-t4 time:"<<(double)(t5-t4)/(CLOCKS_PER_SEC)<<"s"<<" (doInference 1) "<<std::endl;
	std::cout<<"t6-t5 time:"<<(double)(t6-t5)/(CLOCKS_PER_SEC)<<"s"<<" (doInference 2) "<<std::endl;
	std::cout<<"t7-t6 time:"<<(double)(t7-t6)/(CLOCKS_PER_SEC)<<"s"<<" (destroy and show result) "<<std::endl;
	std::cout<<"t7-t1 time:"<<(double)(t6-t1)/(CLOCKS_PER_SEC)<<"s"<<std::endl;
	return 0;
}


































