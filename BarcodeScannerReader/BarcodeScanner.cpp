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

	return rotated(cv::Rect(input_corners[1], input_corners[3]));
}