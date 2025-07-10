// eyestrip.cpp
#include "eyestrip.hpp"

namespace es {

EyeStrip::EyeStrip(float downscale,
                   int stripH,
                   int detectInterval,
                   int maxLost,
                   int smoothFrames)
    : downscale_(downscale),
      stripH_(stripH),
      detectInterval_(detectInterval),
      maxLost_(maxLost),
      smoothFrames_(smoothFrames),
      tracking_(false),
      frameCount_(0),
      lostCount_(0),
      cropped_(false),
      fps_(0.0)
{
    lastTime_ = std::chrono::high_resolution_clock::now();
}

bool EyeStrip::loadCascade(const std::string& xmlPath) {
    return eyeCascade_.load(xmlPath);
}

void EyeStrip::process(const cv::Mat& frame, cv::Mat& debugOut, cv::Mat& stripOut) {
    // 0) Prepare
    frame.copyTo(debugOut);
    stripOut.release();

    // 1) Downscale & grayscale
    cv::resize(frame, small_, cv::Size(), downscale_, downscale_, cv::INTER_AREA);
    cv::cvtColor(small_, gray_, cv::COLOR_BGR2GRAY);

    // 2) Periodic detect
    if ((frameCount_++ % detectInterval_) == 0) {
        std::vector<cv::Rect> eyes;
        eyeCascade_.detectMultiScale(
            gray_, eyes, 1.1, 5, 0,
            cv::Size(20,15), cv::Size(80,80)
        );
        if (eyes.size() >= 2) {
            // pick two largest
            std::nth_element(
                eyes.begin(), eyes.begin()+2, eyes.end(),
                [](auto &a, auto &b){ return a.area()>b.area(); }
            );
            cv::Rect e1 = eyes[0], e2 = eyes[1];
            if (e1.x > e2.x) std::swap(e1,e2);

            // init trackers
            trackerL_ = cv::TrackerKCF::create();
            trackerR_ = cv::TrackerKCF::create();
            trackerL_->init(small_, e1);
            trackerR_->init(small_, e2);

            tracking_ = true;
            lostCount_ = 0;

            // upscale initial positions
            lastL_ = scaleUp(e1);
            lastR_ = scaleUp(e2);
            history_.clear();
        }
    }

    // 3) Track
    if (tracking_) {
        cv::Rect t1, t2;
        bool ok1 = trackerL_->update(small_, t1);
        bool ok2 = trackerR_->update(small_, t2);
        if (ok1 && ok2) {
            lostCount_ = 0;
            lastL_ = scaleUp(t1);
            lastR_ = scaleUp(t2);
        }
        else if (++lostCount_ > maxLost_) {
            tracking_ = false;
            history_.clear();
            cropped_ = false;
        }
    }

    // 4) Build history of union rects
    if (lastL_.area() && lastR_.area()) {
        int y0 = std::min(lastL_.y, lastR_.y);
        int y1 = std::max(lastL_.y+lastL_.height, lastR_.y+lastR_.height);
        int x0 = std::min(lastL_.x, lastR_.x);
        int x1 = std::max(lastL_.x+lastL_.width, lastR_.x+lastR_.width);
        history_.push_back({x0, y0, x1-x0, y1-y0});
        if ((int)history_.size() > smoothFrames_) history_.pop_front();
    }

    // 5) Average history
    if (!history_.empty()) {
        cv::Rect acc(0,0,0,0);
        for (auto &r : history_) {
            acc.x += r.x;    acc.y += r.y;
            acc.width  += r.width;  acc.height += r.height;
        }
        int n = (int)history_.size();
        avg_ = { acc.x/n, acc.y/n, acc.width/n, acc.height/n };
    }

    // 6) Extract strip if we have two eyes
    if (lastL_.area() && lastR_.area()) {
        if (extract(frame, stripOut)) {
            // draw the four strip corners
            for (int i = 0; i < 4; ++i) {
                cv::line(debugOut, corners_[i], 
                         corners_[(i+1)%4], cv::Scalar(0,165,255), 2);
            }
        }
    }

    // 7) Draw debug boxes
    if (cropped_) {
        cv::rectangle(debugOut, crop_, {0,255,0}, 2);
    }
    if (lastL_.area()) {
        cv::rectangle(debugOut, lastL_, {255,0,0}, 2);
        cv::rectangle(debugOut, lastR_, {255,0,0}, 2);
    }

    // 8) Draw FPS
    auto now = std::chrono::high_resolution_clock::now();
    double dt = std::chrono::duration<double>(now - lastTime_).count();
    lastTime_ = now;
    fps_ = 0.9*fps_ + 0.1*(1.0/dt);
    cv::putText(debugOut, cv::format("FPS: %.1f", fps_),
                {10,20}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {0,255,0}, 1);
}

void EyeStrip::reset() {
    tracking_   = false;
    history_.clear();
    cropped_    = false;
}

cv::Rect EyeStrip::scaleUp(const cv::Rect& r) {
    return {
        int(r.x / downscale_),
        int(r.y / downscale_),
        int(r.width  / downscale_),
        int(r.height / downscale_)
    };
}

bool EyeStrip::extract(const cv::Mat& frame, cv::Mat& out) {
    if (lastL_.area() == 0 || lastR_.area() == 0)
        return false;

    // Compute eye centers
    cv::Point2f c1(
        lastL_.x + lastL_.width*0.5f,
        lastL_.y + lastL_.height*0.5f
    );
    cv::Point2f c2(
        lastR_.x + lastR_.width*0.5f,
        lastR_.y + lastR_.height*0.5f
    );
    cv::Point2f ctr = (c1 + c2)*0.5f;

    // Orientation & size
    float dx = c2.x - c1.x, dy = c2.y - c1.y;
    float angle = std::atan2(dy, dx);
    float eyeDist = std::sqrt(dx*dx + dy*dy);
    float stripW = eyeDist * 1.8f;
    float stripH = stripW * (float(stripH_) / 128.0f);

    // Unit vectors
    cv::Point2f ux(std::cos(angle), std::sin(angle));
    cv::Point2f uy(-ux.y, ux.x);

    // Corners: TL, TR, BR, BL
    cv::Point2f tl = ctr - ux*(stripW/2) - uy*(stripH/2);
    cv::Point2f tr = ctr + ux*(stripW/2) - uy*(stripH/2);
    cv::Point2f bl = ctr - ux*(stripW/2) + uy*(stripH/2);

    cv::Point2f srcTri[3] = { tl, bl, tr };
    cv::Point2f dstTri[3] = {
        {0, 0},
        {0, float(stripH_)},
        {128.0f, 0}
    };

    cv::Mat warpMat = cv::getAffineTransform(srcTri, dstTri);
    cv::warpAffine(
        frame, out, warpMat,
        {128, stripH_},
        cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0,0,0)
    );

    // Store corners for debug
    corners_[0] = tl;
    corners_[1] = tr;
    corners_[2] = ctr + ux*(stripW/2) + uy*(stripH/2);
    corners_[3] = bl;

    // Smart cropping after smoothing
    if ((int)history_.size() == smoothFrames_) {
        int mx = avg_.width;
        int my = avg_.height * 2;
        int x0 = std::max(0, avg_.x - mx);
        int y0 = std::max(0, avg_.y - my);
        int w0 = std::min(frame.cols - x0, avg_.width + mx*2);
        int h0 = std::min(frame.rows - y0, avg_.height + my*2);
        cv::Rect c(x0, y0, w0, h0);

        if (!cropped_ ||
            cv::norm(c.tl() - crop_.tl()) > 20 ||
            std::abs(c.area() - crop_.area()) > 3000) {
            crop_ = c;
            cropped_ = true;
        }
    }

    return true;
}

} // namespace es
