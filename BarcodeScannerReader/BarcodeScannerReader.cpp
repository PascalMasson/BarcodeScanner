// BarcodeScannerReader.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "BarcodeScannerReader.h"
#include <opencv2/highgui.hpp>
#include "BarcodeScanner.h"

int main() {
	Log::init("SCANNER");
	LOG_INFO("starting program");

	auto files = std::vector<std::string>();

	files.push_back("barcode_01");
	//files.push_back("barcode_02");
	files.push_back("barcode_03");

	LOG_INFO("Scanning " + std::to_string(std::size(files)) + " images");

	BarcodeFinder finder;

	for (auto file : files) {
		auto image = cv::imread(".\\images\\" + file + ".jpg");

		if (image.empty()) {
			LOG_ERROR("unable to read " + file + ", going to next");
			continue;
		}
		LOG_INFO("Read " + file + ". now looking for barcode data");
		cv::Mat find_output, extract;

		finder.ScanBarcode(image, find_output, extract);
		if (!find_output.empty()) { imshow("Result " + file, find_output); }
		if (!extract.empty()) { imshow("Extract " + file, extract); }
	}
	cv::waitKey(0);
	return 0;
}
