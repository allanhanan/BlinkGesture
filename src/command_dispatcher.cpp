#include "command_dispatcher.hpp"

#include <QProcess>
#include <QStringList>
#include <QDebug>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cctype>
#include <iostream>

// --- Helpers ---

static std::vector<std::string> splitCommand(const std::string& command) {
    std::vector<std::string> parts;
    std::stringstream ss(command);
    std::string part;
    while (std::getline(ss, part, '+')) {
        part.erase(std::remove_if(part.begin(), part.end(), ::isspace), part.end());
        if (!part.empty()) parts.push_back(part);
    }
    return parts;
}

// --- CommandDispatcher using xdotool ---

CommandDispatcher::CommandDispatcher() {
    ready_ = true;
}

CommandDispatcher::~CommandDispatcher() = default;

bool CommandDispatcher::dispatch(const std::string& command) {
    if (!ready_) return false;

    std::cout << "[Dispatching] Command: " << command << std::endl;

    std::vector<std::string> keys = splitCommand(command);
    if (keys.empty()) return false;

    // Mouse click aliases
    static const std::map<std::string, QString> mouseClickMap = {
        {"LClick", "click 1"},
        {"RClick", "click 3"},
        {"MClick", "click 2"},
    };

    // Keys that must retain casing (as expected by xdotool)
    static const QSet<QString> specialKeys = {
        "Tab", "Return", "Enter", "Escape", "BackSpace", "Delete",
        "Up", "Down", "Left", "Right",
        "Home", "End", "Page_Up", "Page_Down",
        "Insert", "Caps_Lock", "Num_Lock", "Scroll_Lock",
        "F1", "F2", "F3", "F4", "F5", "F6",
        "F7", "F8", "F9", "F10", "F11", "F12"
    };

    // If command is just a mouse click like "LClick"
    if (keys.size() == 1) {
        auto it = mouseClickMap.find(keys[0]);
        if (it != mouseClickMap.end()) {
            QStringList args = QStringList() << it->second.split(' ');
            int exitCode = QProcess::execute("xdotool", args);
            if (exitCode != 0) {
                qWarning() << "xdotool mouse click failed with exit code" << exitCode;
                return false;
            }
            return true;
        }
    }

    // Build key combo for keyboard
    QStringList args;
    args << "key";
    QStringList keyParts;
    
    for (const auto& k : keys) {
        QString qk = QString::fromStdString(k).trimmed();
    
        auto it = std::find_if(specialKeys.begin(), specialKeys.end(),
            [&](const QString& val) {
                return val.compare(qk, Qt::CaseInsensitive) == 0;
            });
    
        if (it != specialKeys.end()) {
            keyParts << *it;
        } else if (qk.length() == 1 ||
                   qk.compare("ctrl", Qt::CaseInsensitive) == 0 ||
                   qk.compare("alt", Qt::CaseInsensitive) == 0 ||
                   qk.compare("shift", Qt::CaseInsensitive) == 0 ||
                   qk.compare("super", Qt::CaseInsensitive) == 0 ||
                   qk.compare("meta", Qt::CaseInsensitive) == 0) {
            keyParts << qk.toLower();
        } else {
            qWarning() << "Unknown key:" << qk;
            return false;
        }
    }
    
    args << keyParts.join("+");
    

    int exitCode = QProcess::execute("xdotool", args);
    if (exitCode != 0) {
        qWarning() << "xdotool key dispatch failed with exit code" << exitCode;
        return false;
    }

    return true;
}
