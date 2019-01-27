// BarcodeScannerReader.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "BarcodeScannerReader.h"
#include <opencv2/highgui.hpp>
#include "BarcodeScanner.h"
#include <imgproc.hpp>

int handleError( int status, const char* func_name,
            const char* err_msg, const char* file_name,
            int line, void* userdata )
{
    //Do nothing -- will suppress console output
    return 0;   //Return value is not used
}

int main() {
	Log::init("SCANNER");
	LOG_INFO("starting program");
	BarcodeFinder finder;

	cv::redirectError(handleError);

	//cv::String path("images/*.jpg"); //select only jpg
	//std::vector<cv::String> fn;
	//std::vector<cv::Mat> data;
	//cv::glob(path, fn, true); // recurse
	//for (size_t k = 0; k < fn.size(); ++k) {
	//	cv::Mat im = cv::imread(fn[k]);
	//	if (im.empty())
	//		continue; //only proceed if sucsessful
	//	// you probably want to do some preprocessing
	//	data.push_back(im);
	//}

	//LOG_INFO("Scanning " + std::to_string(std::size(data)) + " images");

	
	//int i = 0;

	//for (auto image : data) {
	//	LOG_INFO("Read " + std::string(fn[i].c_str()) + ". now looking for barcode data");
	//	std::map<std::string, cv::Mat> results;

	//	try { finder.ScanBarcode(image, results); }
	//	catch (cv::Exception& e) {
	//		const char* err_msg = e.what();
	//		std::cout << "exception caught: " << err_msg << std::endl;
	//	}

	//	std::map<std::string, cv::Mat>::iterator it = results.begin();

	//	while (it != results.end()) {
	//		if (!it->second.empty()) {
	//			imshow(std::string(it->first), it->second);
	//			if (it->first == "unedited") { cv::imwrite("C:/barcodeoutput/result.jpg", it->second); }
	//		}
	//		it++;
	//	}
	//	i++;
	//	cv::waitKey(0);
	//}

	cv::VideoCapture cap(0);
	if (!cap.isOpened())
		return -1;
	cv::Mat image;

	for (;;) {
		cap >> image; // Get a new frame from camera
		

		std::map<std::string, cv::Mat> results;

		try { finder.ScanBarcode(image, results); }
		catch (cv::Exception& e) {
			const char* err_msg = e.what();
			std::cout << "exception caught: " << err_msg << std::endl;
		}
		cv::imshow("Result", results["unedited"]); // Display the new frame
		if (cv::waitKey(30) >= 0)
			break;
	}

	return 0;
}

