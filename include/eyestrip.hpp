// eye_strip.hpp
#ifndef EYESTRIP_HPP
#define EYESTRIP_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/objdetect.hpp>
#include <deque>
#include <chrono>

namespace es {

/**
 * EyeStrip: Detects, tracks, and aligns the eye region.
 * Outputs a debug image with overlays and a 128×stripH eye-strip.
 */
class EyeStrip {
public:
    EyeStrip(float downscale = 0.5f,
             int stripH = 24,
             int detectInterval = 5,
             int maxLost = 10,
             int smoothFrames = 5);

    bool loadCascade(const std::string& xmlPath);

    /**
     * Process one frame.
     * @param frame     BGR input frame.
     * @param debugOut  BGR image with overlays drawn.
     * @param stripOut  128×stripH eye strip (empty until first extract).
     */
    void process(const cv::Mat& frame, cv::Mat& debugOut, cv::Mat& stripOut);

    void reset();

private:
    cv::Rect scaleUp(const cv::Rect& r);
    bool extract(const cv::Mat& frame, cv::Mat& out);

    // parameters
    float downscale_;
    int stripH_, detectInterval_, maxLost_, smoothFrames_;

    // cascade & trackers
    cv::CascadeClassifier eyeCascade_;
    cv::Ptr<cv::Tracker> trackerL_, trackerR_;

    // state
    bool tracking_;
    int frameCount_, lostCount_;
    std::deque<cv::Rect> history_;
    cv::Rect lastL_, lastR_, avg_, crop_;
    bool cropped_;

    // debug drawing
    cv::Point2f corners_[4];

    // timing / FPS
    double fps_;
    std::chrono::high_resolution_clock::time_point lastTime_;

    // scratch mats
    cv::Mat small_, gray_;
};

} // namespace es

#endif // EYESTRIP_HPP
