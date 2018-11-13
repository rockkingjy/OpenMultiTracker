#include "kalmantracker.hpp"

namespace sort
{

void KalmanTracker::init(cv::Rect2f bbox, int width, int height)
{
    image_width_ = width;
    image_height_ = height;
    time_since_update_ = 0;
    hit_streak_ = 0;

    kf_ = cv::KalmanFilter(stateSize_, measSize_, contrSize_, type_);
    state_ = cv::Mat(stateSize_, 1, type_);  // [u,v,s,r,du,dv,ds]
    measure_ = cv::Mat(measSize_, 1, type_); // [u,v,s,r]

    // state
    state_.at<float>(0) = bbox.x + bbox.width / 2.0f;
    state_.at<float>(1) = bbox.y + bbox.height / 2.0f;
    state_.at<float>(2) = (float)bbox.width * (float)bbox.height;
    state_.at<float>(3) = (float)bbox.width / (float)bbox.height;
    state_.at<float>(4) = 0.0f;
    state_.at<float>(5) = 0.0f;
    state_.at<float>(6) = 0.0f;

    // measure
    measure_.setTo(cv::Scalar(0));

    // State Transition Matrix A
    // Note: set dT at each processing step!
    // [ 1 0 0 0 1 0 0 ]
    // [ 0 1 0 0 0 1 0 ]
    // [ 0 0 1 0 0 0 1 ]
    // [ 0 0 0 1 0 0 0 ]
    // [ 0 0 0 0 1 0 0 ]
    // [ 0 0 0 0 0 1 0 ]
    // [ 0 0 0 0 0 0 1 ]
    cv::setIdentity(kf_.transitionMatrix);
    kf_.transitionMatrix.at<float>(4) = 1.0f;
    kf_.transitionMatrix.at<float>(12) = 1.0f;
    kf_.transitionMatrix.at<float>(20) = 1.0f;

    // Measure Matrix H
    // [ 1 0 0 0 0 0 0]
    // [ 0 1 0 0 0 0 0]
    // [ 0 0 1 0 0 0 0]
    // [ 0 0 0 1 0 0 0]
    kf_.measurementMatrix = cv::Mat::zeros(measSize_, stateSize_, type_);
    kf_.measurementMatrix.at<float>(0) = 1.0f;
    kf_.measurementMatrix.at<float>(8) = 1.0f;
    kf_.measurementMatrix.at<float>(16) = 1.0f;
    kf_.measurementMatrix.at<float>(24) = 1.0f;

    // Measures Noise Covariance Matrix R
    // [1 0 0  0]
    // [0 1 0  0]
    // [0 0 10 0]
    // [0 0 0 10]
    cv::setIdentity(kf_.measurementNoiseCov, cv::Scalar(1.0f));
    kf_.measurementNoiseCov.at<float>(10) = 10.0f;
    kf_.measurementNoiseCov.at<float>(15) = 10.0f;

    // Posteriori error estimate covariance matrix (P(k))
    kf_.errorCovPost.at<float>(0) = 10.0f;
    kf_.errorCovPost.at<float>(8) = 10.0f;
    kf_.errorCovPost.at<float>(16) = 10.0f;
    kf_.errorCovPost.at<float>(24) = 10.0f;
    kf_.errorCovPost.at<float>(32) = 10000.0f;
    kf_.errorCovPost.at<float>(40) = 10000.0f;
    kf_.errorCovPost.at<float>(48) = 10000.0f;

    // Process Noise Covariance Matrix Q
    kf_.processNoiseCov.at<float>(0) = 1.0f;
    kf_.processNoiseCov.at<float>(8) = 1.0f;
    kf_.processNoiseCov.at<float>(16) = 1.0f;
    kf_.processNoiseCov.at<float>(24) = 1.0f;
    kf_.processNoiseCov.at<float>(32) = 1e-2;
    kf_.processNoiseCov.at<float>(40) = 1e-2;
    kf_.processNoiseCov.at<float>(48) = 1e-4;

    kf_.statePost = state_;
}

void KalmanTracker::update(cv::Rect2f bbox)
{
    time_since_update_ = 0;
    ++hit_streak_;
    measure_.at<float>(0) = bbox.x + bbox.width / 2.0f;
    measure_.at<float>(1) = bbox.y + bbox.height / 2.0f;
    measure_.at<float>(2) = (float)bbox.width * (float)bbox.height;
    measure_.at<float>(3) = (float)bbox.width / (float)bbox.height;

    kf_.correct(measure_); // Kalman Correction
    //cout << "Measure matrix:" << endl << measure_ << endl;
}

void KalmanTracker::predict()
{
    if (time_since_update_ > 0)
    {
        hit_streak_ = 0;
    }
    ++time_since_update_;
    if (outofbound())
    {
        state_.at<float>(0) = 0.0f;
        state_.at<float>(1) = 0.0f;
        state_.at<float>(2) = 0.0f;
        state_.at<float>(3) = 0.0f;
        state_.at<float>(4) = 0.0f;
        state_.at<float>(5) = 0.0f;
        state_.at<float>(6) = 0.0f;
    }
    state_ = kf_.predict();
    //cout << "State post:" << endl << state_ << endl;
}

cv::Rect2f KalmanTracker::get_state()
{
    cv::Rect2f res;
    res.width = std::sqrt(state_.at<float>(2) * state_.at<float>(3));
    res.height = state_.at<float>(2) / res.width;
    res.x = state_.at<float>(0) - res.width / 2.0f;
    res.y = state_.at<float>(1) - res.height / 2.0f;
    return res;
}

bool KalmanTracker::outofbound()
{
    cv::Rect2f res = get_state();
    if (isnan(res.x) | isnan(res.y) | isnan(res.width) | isnan(res.height))
    {
        return true;
    }
    if (res.x + res.width <= 0)
    {
        return true;
    }
    if (res.y + res.height <= 0)
    {
        return true;
    }
    if (res.x >= image_width_)
    {
        return true;
    }
    if (res.y >= image_height_)
    {
        return true;
    }
    if (res.width <= 0)
    {
        return true;
    }
    if (res.height <= 0)
    {
        return true;
    }
    return false;
}

} // namespace sort
