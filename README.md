# IA 2.0 Support by API Google

A robust chatbot built using Python, PyTorch, and integrated with Google's API for enhanced functionality.

## Features
- Natural language processing and text analysis
- Google API integration for advanced responses
- Command-line interface (CLI) for environments without GUI support
- Improved model training with proper text preprocessing
- Comprehensive error handling and fallback mechanisms
- Customizable and extendable

## Installation
1. Clone the repository
    ```sh
    git clone https://github.com/NomadCoderCL/IA_2.0_support_by_API_Google.git
    ```
2. Navigate to the project directory
    ```sh
    cd IA_2.0_support_by_API_Google
    ```
3. Install the required dependencies
    ```sh
    pip install torch torchvision  # Core ML dependencies
    pip install PyQt5             # For GUI (optional)
    pip install google-generativeai  # For Google API (optional)
    ```

## Usage

### Command Line Interface (Recommended)
For a simple command-line interface that works without additional GUI dependencies:
```sh
python cli_app.py
```

### GUI Application
If you have PyQt5 installed, you can run the GUI version:
```sh
python app.py
```

### Training the Model
To train the model on logged interactions:
```sh
python train_from_log.py
```

### Testing the Model
To test the trained model:
```sh
python tests/test_model.py
```

## Recent Improvements

### 🎯 Critical Fixes
- **Fixed training data format mismatch**: Corrected character-level encoding to work properly with the LSTM model
- **Added missing dataset module**: Created proper dataset handling for testing and training
- **Fixed import structure**: Resolved circular dependencies and missing module issues
- **Updated model configuration**: Changed INPUT_SIZE from 784 to 100 for appropriate text processing

### 🚀 Enhanced Features
- **Graceful fallback handling**: App works even when PyQt5 or Google API dependencies are missing
- **Command-line interface**: Added CLI version for environments without GUI support
- **Improved error handling**: Added comprehensive error handling throughout the codebase
- **Better text preprocessing**: Implemented proper character-level encoding with padding
- **Enhanced model architecture**: Added dropout and better LSTM configuration

### 🛡️ Reliability Improvements
- **Input validation**: Added validation for empty inputs and edge cases
- **Robust file handling**: Improved interaction logging with error handling
- **Dependency management**: Clear messaging when optional dependencies are missing
- **Consistent configuration**: Centralized configuration with appropriate defaults

## Project Structure
```
IA_2.0_support_by_API_Google/
├── app.py              # GUI application (requires PyQt5)
├── cli_app.py          # Command-line interface
├── train_from_log.py   # Model training script
├── config.py           # Configuration settings
├── dataset.py          # Dataset handling for training/testing
├── google_api.py       # Google API integration
├── models/
│   └── model.py        # PyTorch model definition
├── utils/
│   └── utils.py        # Utility functions
├── tests/
│   └── test_model.py   # Model testing
└── interactions_log.json  # Logged conversations
```

## Dependencies

### Required
- Python 3.7+
- PyTorch
- NumPy

### Optional
- PyQt5 (for GUI interface)
- google-generativeai (for Google API integration)
- cryptography (for API key encryption)

## Configuration
Edit `config.py` to customize:
- `INPUT_SIZE`: Maximum sequence length (default: 100)
- `HIDDEN_SIZE`: LSTM hidden layer size (default: 128)
- `OUTPUT_SIZE`: Number of output classes (default: 10)
- `IA_NAME`: Chatbot name (default: "NOVA")
