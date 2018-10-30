#include "stdafx.h"
#include "BarcodeScanner.h"
#include <opencv2/imgproc.hpp>

BarcodeFinder::BarcodeFinder() = default;

void BarcodeFinder::ScanBarcode(cv::Mat& input, cv::Mat& output, cv::Mat& contourImage) const {

	//save an unedited version of the image for the barcode extract at the end
	cv::Mat unedited = input.clone();

	//convert image to greyscale
	cv::Mat grey;
	cvtColor(input, grey, CV_BGR2GRAY);

	cv::Mat filtered = applyFilters(grey);

	//Find the largest shape thing in the image
	cv::RotatedRect bounding_rect = findLargestContour(filtered);

	//move the result to the output
	output = input;

	//rotate the barcode to be horizontal
	//and extract it as a new image
	cv::Mat warped = extractBarcodeMat(bounding_rect, unedited, output);

	contourImage = warped;
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

cv::RotatedRect BarcodeFinder::findLargestContour(cv::Mat& filtered) const {
	std::vector<std::vector<cv::Point>> contours;
	findContours(filtered, contours, CV_RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

	if (!contours.empty()) {
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
		///drawContours(filtered, contours, largest_contour_index, cv::Scalar(0, 255, 0), 2);

		return bounding_rect;
	}
	return cv::RotatedRect();
}

cv::Mat BarcodeFinder::extractBarcodeMat(cv::RotatedRect bounding_rect, cv::Mat const unedited,
                                         cv::Mat const output) const {
	//store the corners in an array
	cv::Point2f input_corners[4];
	bounding_rect.points(input_corners);

	//draw a circle at eacht point to indicate the corners (probably remove when odne with debug)
	for (cv::Point2f center : input_corners) { circle(output, center, 5, cv::Scalar(0, 255, 255), 3); }

	//calculate the width and height by using the distance between points
	float const height = distBetweenPoints(input_corners[0], input_corners[1]);
	float const width = distBetweenPoints(input_corners[0], input_corners[3]);

	//use to calculated width and height as dimensions for the
	//new image and create the corner points
	std::vector<cv::Point2f> dest_corners = {
		cv::Point2f(0, 0), cv::Point2f(0, height), cv::Point2f(width, height), cv::Point2f(width, 0)
	};

	//calculate the transformation matrix required to acieve requested transformation
	cv::Mat const mat = getPerspectiveTransform(input_corners, dest_corners.data());

	//execute the transformation
	cv::Mat warped;
	warpPerspective(unedited, warped, mat, cv::Size(width, height));

	//for reading the data it does not matter if image is upside down. 
	//however if the image is in portrait orientation it needs to be rotated 90 degrees
	if (warped.rows > warped.cols) { cv::rotate(warped, warped, cv::ROTATE_90_CLOCKWISE); }

	return warped;
}
