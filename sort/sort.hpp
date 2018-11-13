#ifndef SORT_HPP
#define SORT_HPP

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include "kalmantracker.hpp"
#include "hungarian.hpp"

using namespace std;

namespace sort
{
class Sort
{
public:
  void init(int width, int height);
  void init(int max_age, int min_hits, int width, int height);
  void update(vector<cv::Rect2f> &bbox);
  void associate_detections_to_trackers();
  float iou(cv::Rect2f bbox_gt, cv::Rect2f bb_test);
  void print_trackers();
  void print_detections();
  inline vector<sort::KalmanTracker> get_trackers() const { return trackers_; }

private:
  int image_width_ = 600;
  int image_height_ = 400;
  int max_age_ = 1;
  int min_hits_ = 3;
  int frame_count_ = 0;
  float iou_threhold_ = 0.3;
  vector<sort::KalmanTracker> trackers_;
  vector<cv::Rect2f> detections_;
  vector<int> assignment_;
};
} // namespace sort

#endif