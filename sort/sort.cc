#include "sort.hpp"

namespace sort
{
void Sort::init(int width, int height)
{
    image_width_ = width;
    image_height_ = height;
    trackers_.clear();
    frame_count_ = 0;
}
void Sort::init(int max_age, int min_hits, int width, int height)
{
    image_width_ = width;
    image_height_ = height;
    max_age_ = max_age;
    min_hits_ = min_hits;
    trackers_.clear();
    frame_count_ = 0;
}
//
void Sort::update(vector<cv::Rect2f> &bbox)
{
    ++frame_count_;
    // get the detections.
    detections_.clear();
    for (unsigned int i = 0; i < bbox.size(); i++)
    {
        detections_.push_back(bbox[i]);
    }

    debug();
    print_detections();
    print_trackers();
    // get the predicted locations from existing trackers.
    for (unsigned int i = 0; i < trackers_.size(); i++)
    {
        trackers_[i].predict();
        //debug("inbound: %d", i);
        if (trackers_[i].outofbound())
        {
            //debug("outbound: %d", i);
            trackers_.erase(trackers_.begin() + i);
        }
    }
    debug();
    print_trackers();

    // associate detections and trackers.
    if (!detections_.empty() & !trackers_.empty())
    {
        associate_detections_to_trackers();
        // update matched trackers with assigned detections.
        vector<int> del;
        for (unsigned int i = 0; i < assignment_.size(); i++)
        {
            trackers_[i].update(detections_[assignment_[i]]);
            //debug("%d:%d", i, assignment_[i]);
            del.push_back(assignment_[i]);
        }
        /*        for (unsigned int i = 0; i < del.size(); i++)
        {
            printf("%d ", del[i]);
        }
        debug();*/
        std::sort(del.begin(), del.end(), std::greater<int>());
        for (unsigned int i = 0; i < del.size(); i++)
        {
            detections_.erase(detections_.begin() + del[i]);
        }
    }

    debug();
    print_detections();

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
    // remove aged tracker.
    for (unsigned int i = 0; i < trackers_.size(); i++)
    {
        // remove the tracker when not detected for long time.
        if (trackers_[i].time_since_update_ > max_age_)
        {
            trackers_.erase(trackers_.begin() + i);
        }
    }

    print_trackers();
}
// row: trackers; col: detections.
void Sort::associate_detections_to_trackers()
{
    vector<vector<double>> iou_matrix, iou_matrix_inv;
    for (unsigned int i = 0; i < trackers_.size(); i++)
    {
        iou_matrix.push_back(vector<double>());
        for (unsigned int j = 0; j < detections_.size(); j++)
        {
            iou_matrix[i].push_back(iou(trackers_[i].get_state(), detections_[j]));
        }
    }
    // debug
    debug("%lu x %lu:", iou_matrix.size(), iou_matrix[0].size());
    for (unsigned int i = 0; i < iou_matrix.size(); i++)
    {
        for (unsigned int j = 0; j < iou_matrix[i].size(); j++)
        {
            printf("%f ", iou_matrix[i][j]);
        }
        printf("\n");
    }
    debug();

    // find the max value, to make iou to be >= 0;
    double max = 0;
    for (unsigned int i = 0; i < iou_matrix.size(); i++)
    {
        for (unsigned int j = 0; j < iou_matrix[i].size(); j++)
        {
            if (iou_matrix[i][j] > max)
            {
                max = iou_matrix[i][j];
            }
        }
    }

    // make matrix to be ready for hungraian minum assignment, make sure rows <= cols.
    for (unsigned int i = 0; i < iou_matrix.size(); i++)
    {
        iou_matrix_inv.push_back(vector<double>());
        for (unsigned int j = 0; j < iou_matrix[i].size(); j++)
        {
            iou_matrix_inv[i].push_back(max - iou_matrix[i][j]);
        }
    }
    if (iou_matrix.size() > iou_matrix[0].size())
    {
        for (unsigned int i = iou_matrix[0].size(); i < iou_matrix.size(); i++)
        {
            for (unsigned int j = 0; j < iou_matrix.size(); j++)
            {
                iou_matrix_inv[j].push_back(0.0f);
            }
        }
    }

    // debug
    debug("%lu x %lu:", iou_matrix_inv.size(), iou_matrix_inv[0].size());
    for (unsigned int i = 0; i < iou_matrix_inv.size(); i++)
    {
        for (unsigned int j = 0; j < iou_matrix_inv[i].size(); j++)
        {
            printf("%f ", iou_matrix_inv[i][j]);
        }
        printf("\n");
    }
    debug();

    // assign by hungarian algorithm.
    sort::HungarianAlgorithm hungarian;
    hungarian.Solve(iou_matrix_inv, assignment_);
    for (unsigned int i = 0; i < assignment_.size(); i++)
    {
        std::cout << i << ":" << assignment_[i] << "\t";
    }
    cout << endl;
    debug();

    // remove extra useless assignments when rows <= cols;
    if (iou_matrix.size() > iou_matrix[0].size())
    {
        unsigned int n = assignment_.size();
        for (unsigned int i = iou_matrix[0].size(); i < n; i++)
        {
            assignment_.erase(assignment_.end() - 1);
            /*
    // remove matched with low IOU.
        if (iou_matrix[i][assignment_[i]] < iou_threhold_)
        {
            assignment_.erase(assignment_.begin() + i);
            --i;
        } */
        }
    }
    for (unsigned int i = 0; i < assignment_.size(); i++)
    {
        std::cout << i << ":" << assignment_[i] << "\t";
    }
    cout << endl;
    debug();
}

float Sort::iou(cv::Rect2f bbox_gt, cv::Rect2f bbox_test)
{
    float x1 = (bbox_gt.x > bbox_test.x) ? bbox_gt.x : bbox_test.x;
    float y1 = (bbox_gt.y > bbox_test.y) ? bbox_gt.y : bbox_test.y;
    float x2 = (bbox_gt.x + bbox_gt.width) > (bbox_test.x + bbox_test.width) ? (bbox_test.x + bbox_test.width) : (bbox_gt.x + bbox_gt.width);
    float y2 = (bbox_gt.y + bbox_gt.height) > (bbox_test.y + bbox_test.height) ? (bbox_test.y + bbox_test.height) : (bbox_gt.y + bbox_gt.height);
    float w = x2 - x1 > 0 ? x2 - x1 : 0;
    float h = y2 - y1 > 0 ? y2 - y1 : 0;
    float iou = w * h / ((bbox_gt.area() + bbox_test.area()) - w * h);
    return iou;
}

void Sort::print_trackers()
{
    debug("%lu Trackers:", trackers_.size());
    for (unsigned int i = 0; i < trackers_.size(); i++)
    {
        cv::Rect2f rec = trackers_[i].get_state();
        debug("%f, %f, %f, %f", rec.x, rec.y, rec.width, rec.height);
    }
    debug("-------------------------------");
}
void Sort::print_detections()
{
    debug("%lu detections:", detections_.size());
    for (unsigned int i = 0; i < detections_.size(); i++)
    {
        debug("%f, %f, %f, %f", detections_[i].x, detections_[i].y, detections_[i].width, detections_[i].height);
    }
    debug("-------------------------------");
}

} // namespace sort
