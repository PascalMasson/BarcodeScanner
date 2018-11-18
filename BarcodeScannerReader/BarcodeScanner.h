#pragma once
#include "opencv2/core.hpp"
#include <map>

class BarcodeFinder {
public:
	BarcodeFinder();
	void ScanBarcode(cv::Mat& input, std::map<std::string, cv::Mat>& resultImages) const ;
private:
	cv::Mat applyFilters(cv::Mat& grey) const;
	cv::RotatedRect findLargestContour(cv::Mat& filtered, cv::Mat& unedited) const;
	cv::Mat extractBarcodeMat(cv::RotatedRect bounding_rect, cv::Mat const unedited) const;
	std::vector<int> readBarcodeData(cv::Mat barcodeImage, cv::Mat & output) const;
	int read_digit(const cv::Mat_<uchar> img, cv::Point& cur, std::map<unsigned, char> table, int unit_width, int position) const ;
	void skip_mguard(const cv::Mat_<uchar>& img, cv::Point& cur) const;
	unsigned read_lguard(const cv::Mat_<uchar>& img, cv::Point& cur) const;
	void setup_map(std::map<unsigned, char>& table) const;
	void align_boundary(const cv::Mat_<uchar>& img, cv::Point& cur, int begin, int end) const;
};


