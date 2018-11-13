#ifndef KALMANTRACKER_HPP
#define KALMANTRACKER_HPP

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "debug.hpp"

using namespace std;

namespace sort
{
class KalmanTracker
{
public:
  void init(cv::Rect2f bbox, int width, int height);
  void update(cv::Rect2f bbox);
  void predict();
  cv::Rect2f get_state();
  bool outofbound();

private:
  unsigned int type_ = CV_32F;
  int stateSize_ = 7;
  int measSize_ = 4;
  int contrSize_ = 0;
  cv::KalmanFilter kf_;
  cv::Mat state_;
  cv::Mat measure_;

public:
  int image_width_ = 600;
  int image_height_ = 400;
  int time_since_update_ = 0;
  int hit_streak_ = 0; // how many continous updates. 
};

} // namespace sort

#endif