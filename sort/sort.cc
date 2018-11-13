#include "sort.hpp"

namespace sort
{
void Sort::init(int max_age, int min_hits, int width, int height)
{
    image_width_ = width;
    image_height_ = height;
    max_age_ = max_age;
    min_hits_ = min_hits;
    trackers_.clear();
    frame_count_ = 0;
}

void Sort::update(cv::Rect2f bbox)
{
    ++frame_count_;
    
    // get the predicted locations from existing trackers.
    for (unsigned int i = 0; i < trackers_.size(); i++)
    {
        trackers_[i].predict();
        if (trackers_[i].outofbound())
        {
            trackers_.erase(trackers_.begin() + i);
        }
    }
    // associate detections and trackers.
    if (!detections_.empty())
    {
        associate_detections_to_trackers();
    }
    // update matched trackers with assigned detections.
    for (unsigned int i = 0; i < assignment_.size(); i++)
    {
        trackers_[i].update(detections_[assignment_[i]]);
        detections_.erase(detections_.begin() + i);
    }
    // create and initialise new trackers for unmatched detections.
    if (!detections_.empty())
    {
        for (unsigned int i = 0; i < detections_.size(); i++)
        {
            KalmanTracker kalmantracker;
            kalmantracker.init(detections_[i], image_width_, image_height_);
            trackers_.push_back(kalmantracker);
        }
    }
    // 
    for (unsigned int i = 0; i < trackers_.size(); i++)
    {
        
        // remove the tracker when old enough.
        if(trackers_[i].time_since_update_ > max_age_)
        {
            trackers_.erase(trackers_.begin() + i);
        }
    }

    
}
// row: trackers; col: detections.
void Sort::associate_detections_to_trackers()
{
    vector<vector<double>> iou_matrix;
    for (unsigned int i = 0; i < trackers_.size(); i++)
    {
        iou_matrix.push_back(vector<double>());
        for (unsigned int j = 0; j < detections_.size(); j++)
        {
            iou_matrix[i].push_back(-iou(trackers_[i].get_state(), detections_[j]));
        }
    }
    sort::HungarianAlgorithm hungarian;
    hungarian.Solve(iou_matrix, assignment_);
    for (unsigned int x = 0; x < assignment_.size(); x++)
        std::cout << x << "," << assignment_[x] << "\t";
    debug();
}

float Sort::iou(cv::Rect2f bbox_gt, cv::Rect2f bbox_test)
{
    float x1 = (bbox_gt.x > bbox_test.x) ? bbox_gt.x : bbox_test.x;
    float y1 = (bbox_gt.y > bbox_test.y) ? bbox_gt.y : bbox_test.y;
    float x2 = (bbox_gt.x + bbox_gt.width) > (bbox_test.x + bbox_test.width) ? (bbox_gt.x + bbox_gt.width) : (bbox_test.x + bbox_test.width);
    float y2 = (bbox_gt.y + bbox_gt.height) > (bbox_test.y + bbox_test.height) ? (bbox_gt.y + bbox_gt.height) : (bbox_test.y + bbox_test.height);
    float w = x2 - x1 > 0 ? x2 - x1 : 0;
    float h = y2 - y1 > 0 ? y2 - y1 : 0;
    float iou = w * h / (bbox_gt.area() * bbox_test.area());
    return iou;
}

void Sort::print_trackers_detections()
{
    debug("%lu Trackers:", trackers_.size());
    for (unsigned int i = 0; i < trackers_.size(); i++)
    {
        cv::Rect2f rec = trackers_[i].get_state();
        debug("%f, %f, %f, %f", rec.x, rec.y, rec.width, rec.height);
    }
    debug("%lu detections:", detections_.size());
    for (unsigned int i = 0; i < detections_.size(); i++)
    {
        debug("%f, %f, %f, %f", detections_[i].x, detections_[i].y, detections_[i].width, detections_[i].height);
    }
}

} // namespace sort
