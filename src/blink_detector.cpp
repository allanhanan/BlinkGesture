// blink_detector.cpp
#include "blink_detector.hpp"
#include <cmath>
#include <iostream>

namespace blink {

BlinkDetector::BlinkDetector(const std::string& modelPath,
                             int bufferSize,
                             float closedThreshold,
                             float openThreshold,
                             int blinkFramesMin,
                             int doubleBlinkThreshold)
    : env_(ORT_LOGGING_LEVEL_WARNING, "BlinkDetector"),
      memInfo_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      closedThreshold_(closedThreshold),
      openThreshold_(openThreshold),
      bufferSize_(bufferSize),
      blinkFramesMin_(blinkFramesMin),
      doubleBlinkThreshold_(doubleBlinkThreshold)
{
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);  // single-thread for stability
    session_ = new Ort::Session(env_, modelPath.c_str(), opts);
}

BlinkDetector::~BlinkDetector() {
    delete session_;
}

void BlinkDetector::reset() {
    std::queue<int> empty;
    std::swap(eyeStateBuffer_, empty);
    prevOpen_ = true;
    closedFrames_ = 0;
    currentFrameIdx_ = 0;
    lastBlinkFrame_ = -100;
    doubleBlinkDetected_ = false;
}

void BlinkDetector::updateThresholds(float closedThresh, float openThresh) {
    closedThreshold_ = closedThresh;
    openThreshold_   = openThresh;
}

void BlinkDetector::updateParams(int bufferSize, int blinkFramesMin, int doubleBlinkThreshold) {
    bufferSize_         = bufferSize;
    blinkFramesMin_     = blinkFramesMin;
    doubleBlinkThreshold_ = doubleBlinkThreshold;
}

int BlinkDetector::processStrip(const cv::Mat& eyeStrip) {
    doubleBlinkDetected_ = false;
    if (eyeStrip.empty())
        return UNDEFINED;

    cv::Mat gray = enhanceGray(eyeStrip);
    int pred = scanPatchGrid(gray);

    // Debug
    //std::cout << "[Frame " << currentFrameIdx_ << "] pred = " << pred << std::endl;

    if (pred != UNDEFINED) {
        eyeStateBuffer_.push(pred);
        if ((int)eyeStateBuffer_.size() > bufferSize_)
            eyeStateBuffer_.pop();

        int closedCount = countInQueue(eyeStateBuffer_, CLOSED);
        bool nowOpen = (pred == OPEN && closedCount < blinkFramesMin_);

        if (!nowOpen) {
            ++closedFrames_;
        } else {
            if (!prevOpen_ && closedFrames_ >= blinkFramesMin_) {
                std::cout << "  Blink event (frames closed = " << closedFrames_ << ")\n";
                if (currentFrameIdx_ - lastBlinkFrame_ <= doubleBlinkThreshold_) {
                    doubleBlinkDetected_ = true;
                    std::cout << "    → Double blink!\n";
                    lastBlinkFrame_ = -100;
                } else {
                    lastBlinkFrame_ = currentFrameIdx_;
                }
            }
            closedFrames_ = 0;
        }
        prevOpen_ = nowOpen;
    }
    ++currentFrameIdx_;
    return pred;
}

bool BlinkDetector::didDoubleBlink() const {
    return doubleBlinkDetected_;
}

cv::Mat BlinkDetector::enhanceGray(const cv::Mat& input) {
    cv::Mat gray;
    if (input.channels() == 3)
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    else
        gray = input.clone();

    auto clahe = cv::createCLAHE(2.0);
    cv::Mat enhanced;
    clahe->apply(gray, enhanced);
    return enhanced;
}

int BlinkDetector::scanPatchGrid(const cv::Mat& grayStrip) {
    std::vector<float> votes(2, 0.0f);
    int patches = 0;

    for (float scale : {1.0f, 0.75f}) {
        cv::Mat scaled;
        cv::resize(grayStrip, scaled, cv::Size(), scale, scale);

        for (int y = 0; y < scaled.rows; y += STRIDE) {
            for (int x = 0; x < scaled.cols; x += STRIDE) {
                int w = std::min(IMG_SIZE, scaled.cols - x);
                int h = std::min(IMG_SIZE, scaled.rows - y);
                cv::Rect roi(x, y, w, h);
                cv::Mat patch = scaled(roi);
                cv::resize(patch, patch, cv::Size(IMG_SIZE, IMG_SIZE));
                ++patches;

                int p = classifyPatch(patch);
                if (p == UNDEFINED) 
                    continue;

                if (scale == 1.0f)
                    return p;
                votes[p] += (p == OPEN ? OPEN_WEIGHT : 1.0f);
            }
        }
    }

    if (patches == 0) {
        std::cerr << "  [Warning] No patches scanned!\n";
        return UNDEFINED;
    }

    if (votes[CLOSED] == 0 && votes[OPEN] == 0) {
        // std::cout << "  No confident votes (C=" << votes[CLOSED]
        //           << ", O=" << votes[OPEN] << ")\n";
        return UNDEFINED;
    }
    int result = (votes[OPEN] > votes[CLOSED] ? OPEN : CLOSED);
    // std::cout << "  Votes → Closed=" << votes[CLOSED]
    //           << " | Open="   << votes[OPEN]
    //           << " → Pred="  << result << "\n";
    return result;
}

int BlinkDetector::classifyPatch(const cv::Mat& patch) {
    // 1) Grayscale + CLAHE
    cv::Mat gray;
    if (patch.channels() == 3)
        cv::cvtColor(patch, gray, cv::COLOR_BGR2GRAY);
    else
        gray = patch.clone();
    auto clahe = cv::createCLAHE(2.0);
    clahe->apply(gray, gray);

    // 2) Resize to model input
    cv::Mat resized;
    cv::resize(gray, resized, cv::Size(IMG_SIZE, IMG_SIZE));

    // 3) Prepare INT8 tensor
    std::vector<int8_t> inputData(resized.total());
    for (size_t i = 0; i < resized.total(); ++i)
        inputData[i] = static_cast<int8_t>(resized.data[i] - 128);

    std::array<int64_t,4> shape = {1, 1, IMG_SIZE, IMG_SIZE};
    auto inputTensor = Ort::Value::CreateTensor<int8_t>(
        memInfo_, inputData.data(), inputData.size(), shape.data(), shape.size());

    // 4) Run inference
    const char* inNames[]  = {"input"};
    const char* outNames[] = {"output"};
    auto outputs = session_->Run(
        Ort::RunOptions{nullptr},
        inNames, &inputTensor, 1,
        outNames, 1
    );

    float* scores = outputs.front().GetTensorMutableData<float>();
    float e0 = std::exp(scores[0]);
    float e1 = std::exp(scores[1]);
    float sum = e0 + e1;
    float p0 = e0 / sum;
    float p1 = e1 / sum;

    // 5) Debug softmax & thresholds
    // std::cout << "    Scores=["<< scores[0] <<","<< scores[1] <<"] "
    //           << "Probs=[" << p0 <<","<< p1 <<"] ";

    int pred = UNDEFINED;
    if (p0 >= closedThreshold_ && p0 >= p1) {
        pred = CLOSED;
        //std::cout << "→ CLOSED\n";
    }
    else if (p1 >= openThreshold_ && p1 > p0) {
        pred = OPEN;
        //std::cout << "→ OPEN\n";
    }
    else {
        pred = UNDEFINED;
        //std::cout << "→ UNDEF (thresh)\n";
    }
    return pred;
}

int BlinkDetector::countInQueue(const std::queue<int>& q, int value) const {
    std::queue<int> tmp = q;
    int cnt = 0;
    while (!tmp.empty()) {
        if (tmp.front() == value) ++cnt;
        tmp.pop();
    }
    return cnt;
}

}  // namespace blink
