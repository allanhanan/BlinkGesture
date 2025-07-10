#!/bin/bash


# Check if Python is installed
if ! command -v python3 &>/dev/null; then
    echo "Python3 is not installed. Please install Python3."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &>/dev/null; then
    echo "pip3 is not installed. Please install pip3."
    exit 1
fi

# Create a virtual environment if it doesn't exist
VENV_NAME="BlinkGestureVenV"
if [ ! -d "$VENV_NAME" ]; then
    python3 -m venv "$VENV_NAME"
    echo "Created virtual environment: $VENV_NAME."
else
    echo "Using existing virtual environment: $VENV_NAME."
fi

# Activate the virtual environment
source "$VENV_NAME/bin/activate"

# Check and install required Python packages
install_requirements() {
    if [ -f $(dirname "$0")/requirements.txt ]; then
        echo "Checking and installing required Python packages..."
        pip3 install -r $(dirname "$0")/requirements.txt
    else
        echo "requirements.txt not found."
        exit 1
    fi
}

# Install requirements if needed
install_requirements

# Run the Python script
python3 $(dirname "$0")/main.py

# Deactivate the virtual environment
deactivate

echo "app exit"

