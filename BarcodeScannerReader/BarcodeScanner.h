#pragma once
#include "opencv2/core.hpp"

class BarcodeFinder {
public:
	BarcodeFinder();
	void ScanBarcode(cv::Mat& input, cv::Mat& output, cv::Mat& contourImage) const;
private:
	cv::Mat applyFilters(cv::Mat& grey) const;
	cv::RotatedRect findLargestContour(cv::Mat& filtered) const;
	cv::Mat extractBarcodeMat(cv::RotatedRect bounding_rect, cv::Mat const unedited, cv::Mat const output) const;
};
