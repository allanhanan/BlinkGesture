// blink_detector.hpp
#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <queue>
#include <string>

namespace blink {

enum BlinkState : int {
    CLOSED    = 0,
    OPEN      = 1,
    UNDEFINED = -1
};

class BlinkDetector {
public:
    BlinkDetector(const std::string& modelPath,
                  int bufferSize,
                  float closedThreshold,
                  float openThreshold,
                  int blinkFramesMin,
                  int doubleBlinkThreshold);

    ~BlinkDetector();

    /// Reset all internal state
    void reset();

    /// Update thresholds
    void updateThresholds(float closedThresh, float openThresh);

    /// Update temporal parameters
    void updateParams(int bufferSize, int blinkFramesMin, int doubleBlinkThreshold);

    /**
     * Process a single eye-strip image.
     * @param eyeStrip BGR or gray, extracted by EyeStrip.
     * @return CLOSED, OPEN, or UNDEFINED
     */
    int processStrip(const cv::Mat& eyeStrip);

    /// True if a double blink was detected last call
    bool didDoubleBlink() const;

private:
    cv::Mat enhanceGray(const cv::Mat& input);
    int classifyPatch(const cv::Mat& patch);
    int scanPatchGrid(const cv::Mat& grayStrip);

    int countInQueue(const std::queue<int>& q, int value) const;

    // ONNX
    Ort::Env env_;
    Ort::MemoryInfo memInfo_;
    Ort::Session* session_;

    // Thresholds
    float closedThreshold_;
    float openThreshold_;

    // Timing params
    int bufferSize_;
    int blinkFramesMin_;
    int doubleBlinkThreshold_;

    // State
    std::queue<int> eyeStateBuffer_;
    bool prevOpen_ = true;
    int closedFrames_ = 0;
    int currentFrameIdx_ = 0;
    int lastBlinkFrame_ = -100;
    bool doubleBlinkDetected_ = false;

    // Constants
    static constexpr int IMG_SIZE     = 64;
    static constexpr int STRIDE       = 10;
    static constexpr float OPEN_WEIGHT = 2.5f;
};

}  // namespace blink
