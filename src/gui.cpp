// gui.cpp
#include "gui.hpp"
#include <QApplication>
#include <QStyleFactory>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QFormLayout>
#include <QDebug>
#include <QTimer>
#include <QCloseEvent>
#include <QEvent>

static constexpr int TIMER_INTERVAL_MS = 30;

BlinkGui::BlinkGui(QWidget* parent)
  : QWidget(parent),
    detector_("eye_cnn_int8.onnx",  // model path
              6, 0.7f, 0.60f,        // bufferSize, closedTh, openTh
              4, 30)                // blinkFramesMin, doubleBlinkThresh
{
    setupUI();
    applyDarkStyle();
    setupTrayIcon();
    populateCameraList();

    timer_ = new QTimer(this);
    connect(timer_, &QTimer::timeout, this, &BlinkGui::onFrameUpdate);
}

BlinkGui::~BlinkGui() {
    if (running_) cap_.release();
}

void BlinkGui::setupUI() {
    setWindowTitle("blinkGesture");
    resize(600, 400);

    debugLabel_ = new QLabel();
    debugLabel_->setObjectName("debugLabel");
    debugLabel_->setMinimumSize(480, 270);
    debugLabel_->setAlignment(Qt::AlignCenter);

    // --- Parameter controls ---
    cameraCombo_       = new QComboBox();
    cascadePathEdit_   = new QLineEdit("haarcascade_eye.xml");
    bufferSizeSpin_    = new QSpinBox();  bufferSizeSpin_->setRange(1, 30);  bufferSizeSpin_->setValue(6);
    blinkFramesSpin_   = new QSpinBox();  blinkFramesSpin_->setRange(1, 10); blinkFramesSpin_->setValue(4);
    doubleBlinkSpin_   = new QSpinBox();  doubleBlinkSpin_->setRange(1, 100); doubleBlinkSpin_->setValue(30);
    openThSpin_        = new QDoubleSpinBox(); openThSpin_->setRange(0.0, 1.0); openThSpin_->setSingleStep(0.01); openThSpin_->setValue(0.80);
    closedThSpin_      = new QDoubleSpinBox(); closedThSpin_->setRange(0.0, 1.0); closedThSpin_->setSingleStep(0.01); closedThSpin_->setValue(0.70);
    commandEdit_       = new QLineEdit("Tab");
    startStopBtn_      = new QPushButton("Start");

    QFormLayout* form = new QFormLayout();
    form->addRow("Camera:",         cameraCombo_);
    form->addRow("Cascade:",       cascadePathEdit_);
    form->addRow("Buffer Size:",   bufferSizeSpin_);
    form->addRow("Min Blink:",     blinkFramesSpin_);
    form->addRow("Double Gap:",    doubleBlinkSpin_);
    form->addRow("Open Thresh:",   openThSpin_);
    form->addRow("Closed Thresh:", closedThSpin_);
    form->addRow("Command:",       commandEdit_);

    QVBoxLayout* rightLayout = new QVBoxLayout();
    rightLayout->addLayout(form);
    rightLayout->addStretch();
    rightLayout->addWidget(startStopBtn_, 0, Qt::AlignRight);

    QHBoxLayout* mainLayout = new QHBoxLayout(this);
    mainLayout->addWidget(debugLabel_, 3);
    mainLayout->addLayout(rightLayout, 2);
    mainLayout->setSpacing(20);
    mainLayout->setContentsMargins(15, 15, 15, 15);

    connect(startStopBtn_, &QPushButton::clicked, this, &BlinkGui::onStartStopClicked);
}

void BlinkGui::applyDarkStyle() {
    QApplication::setStyle(QStyleFactory::create("Fusion"));
    static const char* obsStyle = R"(
        QWidget { background: #1e1e1e; color: #cccccc; }
        QLabel#videoLabel, QLabel#debugLabel {
            border: 1px solid #444; background: #000;
        }
        QComboBox, QLineEdit, QSpinBox, QPushButton {
            background: #2d2d2d; border: 1px solid #444;
            padding: 6px; border-radius: 4px; min-width: 80px;
        }
        QPushButton:hover { background: #3e3e3e; }
        QPushButton:pressed { background: #555555; }
        QFormLayout { spacing: 6px; }
        QLabel { font-size: 14px; }
    )";
    setStyleSheet(obsStyle);
}

void BlinkGui::populateCameraList() {
    for (int i = 0; i < 5; ++i) {
        cv::VideoCapture test(i);
        if (test.isOpened()) {
            cameraCombo_->addItem(QString("Cam %1").arg(i), i);
            test.release();
        }
    }
}

void BlinkGui::setupTrayIcon() {
    trayIcon_ = new QSystemTrayIcon(this);
    trayIcon_->setIcon(QIcon::fromTheme("camera"));

    trayMenu_ = new QMenu(this);
    QAction* restoreAction = trayMenu_->addAction("Restore");
    QAction* quitAction = trayMenu_->addAction("Quit");

    connect(restoreAction, &QAction::triggered, this, &BlinkGui::onRestoreFromTray);
    connect(quitAction, &QAction::triggered, this, &BlinkGui::onQuitFromTray);
    connect(trayIcon_, &QSystemTrayIcon::activated, this, &BlinkGui::onTrayIconActivated);

    trayIcon_->setContextMenu(trayMenu_);
    trayIcon_->show();
}

void BlinkGui::closeEvent(QCloseEvent* event) {
    // Let the app close normally when the close button is pressed
    //if (trayIcon_) trayIcon_->hide(); optional: hide tray icon when quitting
    event->accept();
}

void BlinkGui::changeEvent(QEvent* event) {
    QWidget::changeEvent(event);
    if (event->type() == QEvent::WindowStateChange && isMinimized()) {
        QTimer::singleShot(0, this, &QWidget::hide); // hide only on minimize
    }
}


void BlinkGui::onTrayIconActivated(QSystemTrayIcon::ActivationReason reason) {
    if (reason == QSystemTrayIcon::Trigger) {
        showNormal();
        raise();
        activateWindow();
    }
}

void BlinkGui::onRestoreFromTray() {
    showNormal();
    raise();
    activateWindow();
}

void BlinkGui::onQuitFromTray() {
    trayIcon_->hide();
    QApplication::quit();
}

void BlinkGui::onStartStopClicked() {
    if (!running_) {
        int idx = cameraCombo_->currentData().toInt();
        if (!cap_.open(idx)) {
            qWarning() << "Cannot open camera" << idx;
            return;
        }
        if (!eyeStrip_.loadCascade(cascadePathEdit_->text().toStdString())) {
            qWarning() << "Failed to load cascade";
            return;
        }
        detector_.updateParams(
            bufferSizeSpin_->value(),
            blinkFramesSpin_->value(),
            doubleBlinkSpin_->value()
        );
        detector_.updateThresholds(
            closedThSpin_->value(),
            openThSpin_->value()
        );

        dispatcher_ = CommandDispatcher();

        running_ = true;
        startStopBtn_->setText("Stop");
        detector_.reset();
        timer_->start(TIMER_INTERVAL_MS);
    } else {
        timer_->stop();
        cap_.release();
        running_ = false;
        startStopBtn_->setText("Start");
    }
}

void BlinkGui::onFrameUpdate() {
    cv::Mat frame;
    cap_ >> frame;
    if (frame.empty()) return;

    cv::Mat debugImg, strip;
    eyeStrip_.process(frame, debugImg, strip);

    if (!debugImg.empty()) {
        QImage dbg(
            debugImg.data, debugImg.cols, debugImg.rows,
            debugImg.step, QImage::Format_BGR888
        );
        debugLabel_->setPixmap(
            QPixmap::fromImage(dbg)
                  .scaled(debugLabel_->size(),
                          Qt::KeepAspectRatio, Qt::SmoothTransformation)
        );
    }

    if (!strip.empty()) {
        detector_.processStrip(strip);
        if (detector_.didDoubleBlink()) {
            dispatcher_.dispatch(commandEdit_->text().toStdString());
        }
    }
}
