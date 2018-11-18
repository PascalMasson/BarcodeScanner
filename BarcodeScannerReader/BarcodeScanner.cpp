//INCLUDES:
#include "stdafx.h"
#include "BarcodeScanner.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/photo.hpp>
#include <map>

//DEFINES:
#define Ob(x)  ((unsigned)Ob_(0 ## x ## uL))
#define Ob_(x) (x & 1 | x >> 2 & 2 | x >> 4 & 4 | x >> 6 & 8 |		\
	x >> 8 & 16 | x >> 10 & 32 | x >> 12 & 64 | x >> 14 & 128)

#define SPACE 0
#define BAR 255

//TEMPLATES
using pattern_map = std::map<unsigned, char>;

//ENUMS
enum position {
	LEFT,
	RIGHT
};

//CONSTRUCTORS
BarcodeFinder::BarcodeFinder() = default;


void BarcodeFinder::ScanBarcode(cv::Mat& input, std::map<std::string, cv::Mat>& resultImages) const {

	//save an unedited version of the image for the barcode extract at the end
	cv::Mat unedited = input.clone();
	resultImages.insert(std::make_pair("unedited", unedited));
	LOG_TRACE("Image size is " + std::to_string(input.rows) + " by " + std::to_string(input.cols));

	//convert image to greyscale
	cv::Mat grey;
	cvtColor(input, grey, CV_BGR2GRAY);

	cv::Mat filtered = applyFilters(grey);
	resultImages.insert(std::make_pair("filtered", filtered));

	//Find the largest shape thing in the image
	cv::RotatedRect bounding_rect = findLargestContour(filtered, unedited);

	//move the result to the output
	resultImages.insert(std::make_pair("contour", filtered));

	//rotate the barcode to be horizontal
	//and extract it as a new image
	cv::Mat rotated = extractBarcodeMat(bounding_rect, grey);

	resultImages.insert(std::make_pair("rotated", rotated));
	readBarcodeData(rotated, resultImages["rotated"]);
}

cv::Mat BarcodeFinder::applyFilters(cv::Mat& grey) const {

	//Detect horizontal and vertical pixel intensity jumps
	//This indicates an edge (https://docs.opencv.org/3.4/d2/d2c/tutorial_sobel_derivatives.html)
	cv::Mat grad_x, grad_y;
	Sobel(grey, grad_x, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
	Sobel(grey, grad_y, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);

	//convert back to CV_8U
	cv::Mat abs_grad_x, abs_grad_y;
	convertScaleAbs(grad_x, abs_grad_x);
	convertScaleAbs(grad_y, abs_grad_y);

	//combine the horizontal and vertical gradients
	cv::Mat grad;
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
	convertScaleAbs(grad, grad);

	//Blur the input and threshold
	cv::Mat blurred;
	blur(grad, blurred, cv::Point(9, 9));

	cv::Mat tresh;
	threshold(blurred, tresh, 70, 255, cv::THRESH_BINARY);

	//remove the smallet white dots in a sea of black
	cv::Mat closed;
	morphologyEx(tresh, closed, cv::MORPH_CLOSE, getStructuringElement(cv::MORPH_RECT, cv::Point(21, 7)));

	//do a series of erosions and dilations to remove larger white space
	cv::Mat const kern = getStructuringElement(cv::MORPH_RECT, cv::Point(3, 3));
	erode(closed, closed, kern, cv::Point(-1, -1), 4, cv::BORDER_ISOLATED);
	dilate(closed, closed, kern, cv::Point(-1, -1), 4, cv::BORDER_ISOLATED);

	return closed;
}

cv::RotatedRect BarcodeFinder::findLargestContour(cv::Mat& filtered, cv::Mat& unedited) const {
	std::vector<std::vector<cv::Point>> contours;
	findContours(filtered, contours, CV_RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

	if (!contours.empty()) {
		LOG_INFO("Found " + std::to_string(std::size(contours)) + " contours");
		//loop trough contours to find the one with the largest area
		int largest_area = 0, largest_contour_index = 0;
		cv::RotatedRect bounding_rect;
		for (int i = 0; i < contours.size(); i++) // iterate through each contour.
		{
			double const area = contourArea(contours[i]); //  Find the area of contour

			if (area > largest_area) {
				largest_area = area;
				largest_contour_index = i; //Store the index of largest contour
				bounding_rect = minAreaRect(contours[i]); // Find the bounding rectangle for biggest contour
			}
		}

		/// Draw the largest contour using previously stored index.
		drawContours(unedited, contours, largest_contour_index, cv::Scalar(0, 255, 0), 2);

		return bounding_rect;
	}
	throw std::logic_error("No contours found");
}

cv::Mat BarcodeFinder::extractBarcodeMat(cv::RotatedRect bounding_rect, cv::Mat const unedited) const {
	//store the corners in an array
	cv::Point2f input_corners[4];
	bounding_rect.points(input_corners);

	// rect is the RotatedRect (I got it from a contour...)
	// matrices we'll use
	cv::Mat M, rotated, cropped;
	// get angle and size from the bounding box
	float angle = bounding_rect.angle;
	cv::Size rect_size = bounding_rect.size;
	// thanks to http://felix.abecassis.me/2011/10/opencv-rotation-deskewing/
	if (bounding_rect.angle < -45.) {
		angle += 90.0;
		cv::swap(rect_size.width, rect_size.height);
	}
	// get the rotation matrix
	M = cv::getRotationMatrix2D(bounding_rect.center, angle, 1.0);
	// perform the affine transformation
	cv::warpAffine(unedited, rotated, M, unedited.size(), cv::INTER_CUBIC);
	// crop the resulting image
	cv::getRectSubPix(rotated, rect_size, bounding_rect.center, cropped);

	cv::Mat result = rotated(cv::Rect(input_corners[1], input_corners[3]));
	LOG_TRACE("Extracted barcode matrix");
	return result;
}

std::vector<int> BarcodeFinder::readBarcodeData(cv::Mat barcodeImage, cv::Mat& output) const {
	LOG_TRACE("Converting to bw original image has " + std::to_string(barcodeImage.channels()) + " channels");
	cv::Mat_<uchar> img;
	if (barcodeImage.channels() > 1) { cv::cvtColor(barcodeImage, img, cv::COLOR_BGR2GRAY); }
	else { img = barcodeImage; }
	output = img;

	LOG_TRACE("Thresholding");
	//threshold the image
	cv::threshold(img, img, 120, 255, cv::THRESH_BINARY);
	//cv::resize(img, img, cv::Size(), 2, 1);
	cv::threshold(img, img, 120, 255, cv::THRESH_BINARY);

	cv::bitwise_not(img, img);

	LOG_TRACE("Extract is " + std::to_string(img.cols) + " wide");

	pattern_map table;
	setup_map(table);

	output = img;

	cv::Point cur(0, img.rows / 2);

	//skip left quiet zone
	int skipped = 0;
	while (img(cur) == 0) {
		cur.x++;
		skipped++;
	}

	LOG_INFO("Skipped " + std::to_string(skipped) + " colums");
	LOG_INFO("Current pixel value: " + std::to_string(img(cur)));

	int quietwidth = read_lguard(img, cur);

	std::vector<int> digits;
	for (int i = 0; i < 6; ++i) {
		auto digit = read_digit(img, cur, table, quietwidth, LEFT);
		digits.push_back(digit);
	}

	skip_mguard(img, cur);

	for (int i = 0; i < 6; ++i) { digits.push_back(read_digit(img, cur, table, quietwidth, RIGHT)); }

	std::string digit_string = "";
	for (int i = 0; i < std::size(digits); ++i)
		digit_string = digit_string + std::to_string(digits[i]) + " ";

	LOG_INFO("Read digits: [ " + digit_string + "]");

	return std::vector<int>();
}

int BarcodeFinder::read_digit(const cv::Mat_<uchar> img, cv::Point& cur, pattern_map table, int unit_width,
                              int position) const {
	int pattern[7] = {0, 0, 0, 0, 0, 0, 0};
	for (int i = 0; i < 7; i++) {
		LOG_TRACE("READING DIGIT BAR" + std::to_string(i) + " CURRENT POSISTION IS: " + std::to_string(cur.x));
		//loop through all pixls thhat should be in this bit-bar
		for (int j = 0; j < unit_width; j++) {
			if (img(cur) == 255)
				++pattern[i];
			LOG_TRACE("moving forward");
			++cur.x;
			if(cur.x > img.cols) {
				throw std::out_of_range("point is ouside of image");
			}
		}


		if (pattern[i] == 1 && img(cur) == BAR
			|| pattern[i] == unit_width - 1 && img(cur) == SPACE) {
			--cur.x;
			LOG_TRACE("Correcting");
		}
	}

	LOG_TRACE("Found pattern: " + std::to_string(pattern[0])+ std::to_string(pattern[1])+ std::to_string(pattern[2])+
		std
		::to_string(pattern[3])+ std::to_string(pattern[4])+ std::to_string(pattern[5])+ std::to_string(pattern[6]));

	//cosntruct the 7 bit number for the read digit
	float converstion_threshold = ((float)unit_width) / 2;
	unsigned v = 0;
	for (int i = 0; i < 7; i++) { v = (v << 1) + (pattern[i] >= converstion_threshold); }

	//look up the value for the 7bit number
	char digit;
	if (position == LEFT) {
		if (table.count(v))
			digit = table[v];
		else
			digit = -1;
		align_boundary(img, cur, SPACE, BAR);
	}
	else {
		if (table.count(~v & Ob(1111111)))
			digit = table[~v & Ob(1111111)];
		else
			digit = -1;
		align_boundary(img, cur, BAR, SPACE);
	}
	LOG_TRACE("We read: " + std::to_string(digit));
	return digit;
}

void BarcodeFinder::skip_mguard(const cv::Mat_<uchar>& img, cv::Point& cur) const {
	int pattern[5] = {SPACE, BAR, SPACE, BAR, SPACE};
	for (int i = 0; i < 5; ++i)
		while (img(cur) == pattern[i])
			++cur.x;
}

unsigned BarcodeFinder::read_lguard(const cv::Mat_<uchar>& img, cv::Point& cur) const {
	int widths[3] = {0, 0, 0};
	int pattern[3] = {BAR, SPACE, BAR};
	for (int i = 0; i < 3; ++i)
		while (img(cur) == pattern[i]) {
			++cur.x;
			++widths[i];
		}

	LOG_INFO(std::to_string(widths[0]) + " " + std::to_string(widths[1]) + " " + std::to_string(widths[2]) + " ");

	return widths[0];
}

void BarcodeFinder::setup_map(pattern_map& table) const {
	table.insert(std::make_pair(Ob(0001101), 0));
	table.insert(std::make_pair(Ob(0011001), 1));
	table.insert(std::make_pair(Ob(0010011), 2));
	table.insert(std::make_pair(Ob(0111101), 3));
	table.insert(std::make_pair(Ob(0100011), 4));
	table.insert(std::make_pair(Ob(0110001), 5));
	table.insert(std::make_pair(Ob(0101111), 6));
	table.insert(std::make_pair(Ob(0111011), 7));
	table.insert(std::make_pair(Ob(0110111), 8));
	table.insert(std::make_pair(Ob(0001011), 9));

	table.insert(std::make_pair(Ob(0100111), 0));
	table.insert(std::make_pair(Ob(0110011), 1));
	table.insert(std::make_pair(Ob(0011011), 2));
	table.insert(std::make_pair(Ob(0100001), 3));
	table.insert(std::make_pair(Ob(0011101), 4));
	table.insert(std::make_pair(Ob(0111001), 5));
	table.insert(std::make_pair(Ob(0000101), 6));
	table.insert(std::make_pair(Ob(0010001), 7));
	table.insert(std::make_pair(Ob(0001001), 8));
	table.insert(std::make_pair(Ob(0010111), 9));
}

void BarcodeFinder::align_boundary(const cv::Mat_<uchar>& img, cv::Point& cur, int begin, int end) const {
	if (img(cur) == end) {
		while (img(cur) == end)
			++cur.x;
	}
	else {
		while (img(cur.y, cur.x - 1) == begin)
			--cur.x;
	}
}
