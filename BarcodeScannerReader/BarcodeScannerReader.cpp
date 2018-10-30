// BarcodeScannerReader.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "BarcodeScannerReader.h"
#include <opencv2/highgui.hpp>
#include "BarcodeScanner.h"

#define Ob(x)  ((unsigned)Ob_(0 ## x ## uL))
#define Ob_(x) (x & 1 | x >> 2 & 2 | x >> 4 & 4 | x >> 6 & 8 |		\
	x >> 8 & 16 | x >> 10 & 32 | x >> 12 & 64 | x >> 14 & 128)

int main() {
	auto files = std::vector<std::string>();

	files.push_back("barcode_01");
	files.push_back("barcode_02");
	files.push_back("barcode_03");
	files.push_back("chessboard");
	files.push_back("qr");
	files.push_back("qr_2");
	files.push_back("zebra");

	BarcodeFinder finder;

	for (auto file : files) {
		auto image = cv::imread(".\\images\\" + file + ".jpg");

		if (image.empty()) {
			std::cout << "Could not open or find the image" << std::endl;
			return -1;
		}

		cv::Mat find_output, extract;

		finder.ScanBarcode(image, find_output, extract);
		if (!find_output.empty()) { imshow("Result " + file, find_output); }
		if (!extract.empty()) {
			//imshow("Extract " + file, extract); 
		}
	}
	cv::waitKey(0);
	return 0;
}
