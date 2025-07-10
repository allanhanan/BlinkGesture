#pragma once

#include <string>

class CommandDispatcher {
public:
    CommandDispatcher();
    ~CommandDispatcher();

    bool dispatch(const std::string& command);

private:
    bool ready_ = false;
};
