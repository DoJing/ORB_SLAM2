//
// Created by dojing on 2021/1/12.
//

#ifndef TRIANGLE_TRIANGLEPLANES_H
#define TRIANGLE_TRIANGLEPLANES_H
#include "Triangle.h"
#include <opencv2/opencv.hpp>
std::vector<Triangle> ComputeDelaunayTriangulation (std::vector<support_pt> p_support);
void ComputePlanes (const std::vector<support_pt>& p_support,std::vector<Triangle> &triangles);
void ComputeDeepth(const std::vector<support_pt>& p_support,const std::vector<Triangle>& tri,cv::Mat& deep);
#endif //TRIANGLE_TRIANGLEPLANES_H
