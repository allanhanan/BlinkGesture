// main.cpp

#include <QApplication>
#include "gui.hpp"

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    BlinkGui gui;
    gui.show();
    return app.exec();
}
