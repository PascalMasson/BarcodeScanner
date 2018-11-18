#include "stdafx.h"
#include "MathHelper.h"
#include "cmath"

float distBetweenPoints(cv::Point2f p, cv::Point2f q) { return std::sqrtf(pow(p.x - q.x, 2) + pow(p.y - q.y, 2)); }
