#ifndef BARCODESCANNERREADER_H
#define BARCODESCANNERREADER_H

#include <opencv2\core.hpp>

void findBarcode(cv::Mat& input, cv::Mat& output, cv::Mat& contourImage);

#endif

