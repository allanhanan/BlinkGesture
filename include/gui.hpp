#pragma once

#include <QWidget>
#include <QTimer>
#include <QComboBox>
#include <QLineEdit>
#include <QSpinBox>
#include <QPushButton>
#include <QLabel>
#include <QSystemTrayIcon>
#include <QMenu>
#include <QCloseEvent>
#include <QEvent>
#include <opencv2/opencv.hpp>

#include "eyestrip.hpp"
#include "blink_detector.hpp"
#include "command_dispatcher.hpp"

class BlinkGui : public QWidget {
    Q_OBJECT

public:
    explicit BlinkGui(QWidget* parent = nullptr);
    ~BlinkGui() override;

protected:
    void closeEvent(QCloseEvent* event) override;
    void changeEvent(QEvent* event) override;

private slots:
    void onStartStopClicked();
    void onFrameUpdate();
    void onTrayIconActivated(QSystemTrayIcon::ActivationReason reason);
    void onRestoreFromTray();
    void onQuitFromTray();

private:
    void setupUI();
    void applyDarkStyle();
    void populateCameraList();
    void setupTrayIcon();

    // pipeline modules
    es::EyeStrip         eyeStrip_;
    blink::BlinkDetector detector_;
    CommandDispatcher    dispatcher_;

    // GUI widgets
    QComboBox*       cameraCombo_;
    QLineEdit*       cascadePathEdit_;
    QLineEdit*       commandEdit_;
    QSpinBox*        bufferSizeSpin_;
    QSpinBox*        blinkFramesSpin_;
    QSpinBox*        doubleBlinkSpin_;
    QPushButton*     startStopBtn_;
    QDoubleSpinBox*  openThSpin_;
    QDoubleSpinBox*  closedThSpin_;
    QLabel*          debugLabel_; 
    QTimer*          timer_;

    
    cv::VideoCapture cap_;
    bool running_ = false;

    QSystemTrayIcon* trayIcon_ = nullptr;
    QMenu*           trayMenu_ = nullptr;
};
