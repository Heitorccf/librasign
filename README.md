<div align="center">

[**English Version 🇺🇸 | 🇬🇧**](#english-version)

[**Versão em Português 🇧🇷 | 🇵🇹**](#versão-em-português)

</div>

---

## 🇺🇸 | 🇬🇧 English Version

# LibraSign

## Table of Contents

  * [❗ Introduction](https://www.google.com/search?q=%23-introduction)
  * [Academic Foundation](https://www.google.com/search?q=%23academic-foundation)
  * [System Overview](https://www.google.com/search?q=%23system-overview)
      * [Scope and Limitations](https://www.google.com/search?q=%23scope-and-limitations)
      * [System Architecture](https://www.google.com/search?q=%23system-architecture)
      * [Processing Workflow](https://www.google.com/search?q=%23processing-workflow)
  * [System Requirements](https://www.google.com/search?q=%23system-requirements)
      * [Software Requirements](https://www.google.com/search?q=%23software-requirements)
      * [Hardware Requirements](https://www.google.com/search?q=%23hardware-requirements)
  * [Installation Guide](https://www.google.com/search?q=%23installation-guide)
      * [Step 1: Environment Preparation](https://www.google.com/search?q=%23step-1-environment-preparation)
      * [Step 2: Clone the Repository](https://www.google.com/search?q=%23step-2-clone-the-repository)
      * [Step 3: Configure the Virtual Environment](https://www.google.com/search?q=%23step-3-configure-the-virtual-environment)
      * [Step 4: Install Dependencies](https://www.google.com/search?q=%23step-4-install-dependencies)
  * [Running the System](https://www.google.com/search?q=%23running-the-system)
      * [Standard Usage Mode](https://www.google.com/search?q=%23standard-usage-mode)
      * [Capturing a New Dataset (Optional)](https://www.google.com/search?q=%23capturing-a-new-dataset-optional)
      * [Retraining the Model](https://www.google.com/search?q=%23retraining-the-model)
  * [Public Dataset](https://www.google.com/search?q=%23public-dataset)
  * [Troubleshooting](https://www.google.com/search?q=%23troubleshooting)
  * [Applicability and Extensibility](https://www.google.com/search?q=%23applicability-and-extensibility)
  * [Final Considerations](https://www.google.com/search?q=%23final-considerations)
  * [References and Complementary Documentation](https://www.google.com/search?q=%23references-and-complementary-documentation)

-----

## ❗ Introduction

**LibraSign** is an open-source system, developed as a undergraduate thesis (*TCC*), that uses computer vision and machine learning techniques to perform real-time recognition of gestures corresponding to the manual alphabet of Brazilian Sign Language (**Libras**). The project was designed to explore methodologies for processing geometric data and neural classification for communication accessibility applications.

It is important to understand that LibraSign has a deliberately restricted scope: the system *exclusively* recognizes the hand configurations corresponding to the letters of the manual alphabet (A to Z). It is not capable of interpreting complete words, complex signs, or the spatial grammar of Libras. This limitation was established to allow for a focused academic investigation into the effectiveness of artificial neural networks in classifying static gestures, serving as a proof of concept for future expansions.

The project is primarily intended for academic and educational environments, serving as a study tool for visual signal processing and supervised learning. Although functional, the system is not designed to replace professional interpreters or for large-scale daily communication, as Libras involves complex linguistic elements beyond the scope of this work, including facial expressions, body movements, and its own grammatical structures.

-----

## Academic Foundation

This project is the materialization of a scientific investigation conducted within the Bachelor's program in Information Systems. The complete theoretical foundation—including a literature review on sign languages, computer vision techniques, neural network architectures, experimental methodology, statistical analysis of results, and discussion on the social implications of assistive technology—is detailed in the final year project document.

For a deeper understanding of the theoretical foundations, architectural decisions, experiments, and conclusions, reading the full academic document, available in this repository, is recommended:

**📄 HeitorFernandes-TCC\_BSI.pdf**

The academic document addresses key topics such as the difference between sign language communication and dactylology (fingerspelling), the limitations of approaches based on raw image processing, the choice of geometric landmark representations, and the performance metrics obtained through stratified cross-validation.

-----

## System Overview

### Scope and Limitations

Before using the system, it is important for the user to clearly understand the functional scope of LibraSign. The system was specifically developed to recognize the static hand configurations of the Libras manual alphabet, corresponding to the twenty-six letters of the Latin alphabet (A–Z). This methodological choice was deliberate and aligns with the project's research objectives.

**What the system recognizes:**

  * Hand configurations corresponding to each of the letters from A to Z of the Libras manual alphabet, when presented statically and in isolation in front of the camera.

**What the system does NOT recognize:**

  * Complete words in Libras, which are often represented by unique signs, not by letter-by-letter spelling.
  * Compound or ideographic signs that make up the standard vocabulary of the language.
  * Facial expressions, body movement, or the use of signing space, which are essential elements of Libras grammar.
  * Regional variations or dialects of the sign language.
  * Dynamic transitions between letters or gestures in continuous motion.

This limitation positions LibraSign as an educational and research tool, suitable for studying pattern recognition techniques and for educational applications in teaching the manual alphabet, but not as a complete sign language translator. The project establishes a foundation that can be expanded in future work to include a broader vocabulary and additional linguistic elements.

### System Architecture

LibraSign was architected using a modular methodology that clearly separates the responsibilities of each system component. This organization facilitates maintenance, testing, and the eventual expansion of functionalities. The architecture comprises three main modules:

  * **Data Capture Module (`src/capture.py`):** This component is responsible for acquiring training data. Using the **MediaPipe** library developed by Google, the module accesses the device's camera and performs real-time detection of hands in the field of view. For each captured frame, MediaPipe identifies twenty-one anatomical reference points (landmarks) on the detected hand, extracting their three-dimensional coordinates (x, y, z) in normalized space. This geometric data, rather than raw pixel images, is persisted in CSV files organized by class, creating a lightweight and structured dataset that facilitates later processing.

  * **Training Module (`src/train.py`):** This component implements the complete supervised learning pipeline. Initially, the module loads the landmark dataset from the CSV files generated during the capture phase. It then applies a geometric normalization transformation that makes the data invariant to the hand's absolute position in the frame and to scale (distance from the camera), centering the points relative to the wrist and normalizing by the hand's characteristic length. After normalization, the data is standardized using `StandardScaler` to have a mean of zero and unit variance. The chosen model is a **Multi-Layer Perceptron (MLP)** with two hidden layers, trained using the backpropagation algorithm. Performance evaluation is conducted using 5-fold stratified cross-validation, ensuring robust estimates of generalization ability. Finally, the trained model, the standardization object, and the class mapping are serialized for use in inference.

  * **Real-Time Prediction Module (`src/predict.py`):** This is the user-facing module, responsible for the practical application of the trained model. The component loads the persisted artifacts (model, scaler, and classes), initializes the video capture, and processes each frame in real-time. For each hand detection, the landmark coordinates are extracted, normalized, and standardized in exactly the same way as during training, ensuring the consistency of the input data. The resulting vector is fed to the neural classifier, which returns probabilities for each class. The system implements two stabilization strategies: a majority vote filter over the last ten frames to smooth out noisy predictions, and a temporal confirmation mechanism that requires a letter to remain stable for two seconds before being added to the constructed sentence. The result is presented visually on the screen, along with confidence indicators and the formed sentence.

### Processing Workflow

To aid in understanding the system's operation, the following is a sequential processing flow from gesture capture to result presentation:

1.  **Frame Acquisition:** The system continuously captures frames from the device's camera in real-time, processing approximately twenty frames per second depending on the available hardware.
2.  **Hand Detection:** Each frame is processed by the MediaPipe hand detection model, which uses lightweight convolutional neural networks to identify the presence and location of hands in the image. When a hand is detected with confidence above the established threshold, the system proceeds to landmark extraction.
3.  **Landmark Extraction:** MediaPipe identifies twenty-one anatomical points on the detected hand, corresponding to locations such as the tip of each finger, the metacarpophalangeal, proximal interphalangeal, and distal interphalangeal joints, as well as the wrist. Each landmark is represented by its three-dimensional coordinates, normalized relative to the image dimensions.
4.  **Geometric Normalization:** The raw landmarks are transformed to ensure invariance. First, all points are translated so that the wrist (landmark zero) is at the origin of the coordinate system. Next, the Euclidean distance between the wrist and the base of the middle finger (landmark nine) is calculated and used as a scaling factor. All points are then divided by this factor, resulting in a representation where the absolute size and position of the hand do not influence the classification.
5.  **Statistical Standardization:** The geometrically normalized data is standardized using the trained `StandardScaler`, which subtracts the mean and divides by the standard deviation for each feature, as calculated on the training set. This step ensures that all dimensions of the input vector contribute equally to the classifier's decision.
6.  **Neural Classification:** The standardized feature vector is propagated through the MLP layers. The network processes the information through its one hundred and twenty-eight units in the first hidden layer, followed by sixty-four units in the second layer, applying non-linear activation functions. The output layer, with dimensionality equal to the number of classes, produces probabilities via a softmax function.
7.  **Temporal Stabilization:** To reduce oscillations and spurious predictions, the system maintains a history of the last ten predictions and applies a majority vote. Additionally, a letter is only considered confirmed if it remains the predominant prediction for two consecutive seconds, preventing transient movements from being interpreted as intentional gestures.
8.  **Result Presentation:** The system renders the detected landmarks, the currently recognized letter with a confidence indicator, a progress bar for temporal confirmation, and, at the bottom of the screen, the sentence formed by the confirmed letters, all overlaid on the video frame.

-----

## System Requirements

### Software Requirements

For the correct execution of LibraSign, the development environment must meet the following software requirements:

  * **Operating System:** The system was developed and tested in Linux (Debian and Fedora-based distributions), macOS (versions 11 and higher), and Windows 10/11 environments. In theory, any operating system that supports Python and the necessary libraries should be able to run the software.
  * **Python Interpreter:** Installation of Python version **3.11.13** is required, as specified during the project's development. Versions prior to 3.9 are not supported. Later versions might work but have not been extensively tested and may present incompatibilities with some dependencies.
  * **pip Package Manager:** Project dependencies are installed via `pip`, the standard Python package manager. Recent versions of Python include `pip`, but it is advisable to check its presence and update it before proceeding.
  * **Functional Camera:** The system requires access to a camera (integrated or external webcam) for real-time video capture. Ensure the operating system has granted the necessary permissions for applications to access the camera.
  * **Essential Python Libraries:** The following libraries form the functional core of the system, and their specific versions are important for proper operation:

<!-- end list -->

```
# Recommended Python version for this project: 3.11.13

scikit-learn==1.7.2
numpy==2.2.6
pandas==2.3.2
opencv-python==4.12.0.88
mediapipe==0.10.14
kagglehub==0.3.13
```

### Hardware Requirements

Although LibraSign has been optimized to run on modest hardware, certain minimum requirements must be met to ensure adequate performance:

  * **Processor:** A modern processor with at least two physical cores operating at a base frequency of 2.0 GHz or higher is recommended. 8th generation Intel Core i3 or higher, AMD Ryzen 3, or equivalent processors are suitable. The system was successfully tested on Intel Core i5 and i7 processors, as well as on ARM processors of Apple Silicon devices.
  * **RAM:** A minimum of 4 GB of RAM is required to run the system. However, 8 GB or more is recommended for comfortable operation, especially during model training, which can consume significant memory depending on the dataset size.
  * **Storage:** The project itself occupies approximately 16.6 MB of disk space. The public landmark dataset, when downloaded, adds about 32.39 MB. It is recommended to have at least 1 GB of free disk space to accommodate the project, datasets, trained models, and any additional datasets the user may wish to capture.
  * **Camera:** A camera with a minimum resolution of 640x480 pixels (VGA) and a capture rate of at least 15 frames per second is essential. Cameras with HD (1280x720) or higher resolution and 30 FPS rates provide a better user experience. The camera should be positioned to clearly capture the user's hand against a relatively uniform background, preferably with good ambient lighting.
  * **Lighting:** While not a hardware requirement per se, adequate lighting conditions are crucial for the system's performance. A well-lit environment is recommended, preferably with diffuse natural light or uniform artificial lighting, avoiding strong backlighting or harsh shadows that could hinder landmark detection by the MediaPipe library.
  * **Graphics System:** Although the system does not require a dedicated GPU, basic support for displaying graphical windows and rendering video is necessary. On Linux systems, ensure the X server (X11) or Wayland is configured correctly. The system will not function properly in server or container environments without a graphical interface.

-----

## Installation Guide

This guide will walk the user through all the necessary steps to prepare the development environment and install LibraSign on their system. The instructions are detailed and include specific commands for different operating systems where applicable.

### Step 1: Environment Preparation

Before starting the LibraSign installation, you must ensure that the Python interpreter is installed and correctly configured on the system.

**Verify Python Installation:**

First, check if Python is installed and which version is available. Open your terminal (Linux/macOS) or Command Prompt/PowerShell (Windows) and run the following command:

```bash
python --version
```

On some systems, especially Linux and macOS, you may need to use the `python3` command explicitly:

```bash
python3 --version
```

The command should return the installed Python version, ideally `Python 3.11.13` or a 3.11.x version. If the returned version is lower than 3.9, you will need to update Python. If the command is not recognized, Python is not installed or not correctly configured in the system's PATH.

**Install Python (if necessary):**

If Python is not installed, follow the specific instructions for your operating system:

  * **Linux (Debian/Ubuntu):**
    ```bash
    sudo apt update
    sudo apt install python3.11 python3.11-venv python3-pip
    ```
  * **Linux (Fedora):**
    ```bash
    sudo dnf install python3.11
    ```
  * **macOS:** It is recommended to use Homebrew for installation:
    ```bash
    brew install python@3.11
    ```
  * **Windows:** Download the official Python 3.11 installer from `python.org`, making sure to check the "**Add Python to PATH**" option during installation.

**Verify pip:**

`pip` should be installed automatically with Python. Check its presence and version:

```bash
python -m pip --version
```

or

```bash
python3 -m pip --version
```

If `pip` is not available, it can be installed via the `get-pip.py` script available on the official Python website.

### Step 2: Clone the Repository

With Python properly installed, the next step is to obtain a local copy of the LibraSign repository. Ensure you have **Git** installed on your system before proceeding.

**Verify Git:**

```bash
git --version
```

If Git is not installed, visit `git-scm.com` and follow the instructions for your operating system.

**Clone the Repository:**

Navigate to the directory where you want to store the project and run the following command to clone the repository:

```bash
git clone https://github.com/heitorccf/librasign.git
```

Wait while Git downloads all the repository files. Upon completion, a new directory named `librasign` will be created containing all project files.

**Navigate to the Project Directory:**

Enter the newly created directory:

```bash
cd librasign
```

All subsequent commands must be executed from this project root directory.

### Step 3: Configure the Virtual Environment

Using a Python virtual environment is a highly recommended practice for Python projects. The virtual environment creates an isolated context where project dependencies can be installed without interfering with other projects or system-wide libraries. This approach prevents version conflicts and facilitates the reproducibility of the execution environment.

**Create the Virtual Environment:**

  * **Linux and macOS:**
    ```bash
    python3 -m venv .venv
    ```
  * **Windows:**
    ```bash
    python -m venv .venv
    ```

This command creates a directory named `.venv` inside the project directory, containing an isolated copy of the Python interpreter and associated tools. The leading dot in the name (`.venv`) is a convention indicating a hidden or configuration directory.

**Activate the Virtual Environment:**

After creating the virtual environment, you need to activate it. The activation command varies by operating system:

  * **Linux and macOS:**
    ```bash
    source .venv/bin/activate
    ```
  * **Windows (Command Prompt):**
    ```bash
    .venv\Scripts\activate.bat
    ```
  * **Windows (PowerShell):**
    ```bash
    .venv\Scripts\Activate.ps1
    ```

*Important note for Windows PowerShell users:* If you encounter an error related to script execution policy, you may need to temporarily change the policy by running PowerShell as an administrator and typing:

```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

After successful activation, you will notice that your terminal prompt has been modified, displaying the virtual environment's name in parentheses, usually `(.venv)`, before the directory path. This visual confirmation indicates that the virtual environment is active and that any packages installed via `pip` will be installed in this isolated environment.

**Deactivate the Virtual Environment (for future reference):**

Although not necessary immediately, it is useful to know that when you wish to exit the virtual environment, you just need to run:

```bash
deactivate
```

This command works on all operating systems and returns the terminal to the system's global Python environment.

### Step 4: Install Dependencies

With the virtual environment activated, proceed to install all the libraries necessary to run LibraSign. The project includes a `requirements.txt` file that lists all dependencies with their specific versions, facilitating installation in a single operation.

**Update pip (recommended):**

Before installing dependencies, it is wise to ensure `pip` is updated to its latest version:

```bash
python -m pip install --upgrade pip
```

**Install Dependencies:**

Run the following command to install all libraries listed in the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

`pip` will process the file, resolve dependencies, download the necessary packages from the official Python Package Index (PyPI) repositories, and install them in the virtual environment. This process may take a few minutes depending on your internet connection speed and system processing power.

During the installation, you will see messages indicating the download and installation progress of each package. It is normal for some packages to show compilation messages, especially on Linux systems, if C extensions need to be compiled. Wait until the process is fully completed.

**Verify Installation:**

After the installation is complete, it is recommended to verify that the main libraries were installed correctly. You can list all installed libraries with:

```bash
pip list
```

Look for the essential libraries in the list: `scikit-learn`, `numpy`, `pandas`, `opencv-python`, `mediapipe`, and `kagglehub`. If all are present, the installation was successful.

Alternatively, you can test the import of the libraries directly in the Python interpreter:

```bash
python -c "import cv2, mediapipe, sklearn, numpy, pandas; print('All libraries were imported successfully')"
```

If the command executes without errors and displays the success message, the environment is correctly configured and ready for use.

-----

## Running the System

Once the environment is properly prepared and all dependencies are installed, the user can proceed to run LibraSign. It is important to understand that the system offers different modes of operation, each serving specific purposes.

### Standard Usage Mode

For most users interested in trying out the gesture recognition system without needing to train a new model, the standard usage mode is the most appropriate. This mode uses a pre-trained model that was developed with the public dataset available on Kaggle.

**Run the Real-Time Translator:**

With the virtual environment activated, run the prediction script:

```bash
python src/predict.py
```

Upon running this command, the system will perform the following operations:

  * First, it will load the trained model artifacts from the `models/` directory, including the serialized MLP classifier, the `StandardScaler` object used for data standardization, and the class mapping that links numerical indices to the alphabet letters.
  * Next, it will initialize the MediaPipe library for hand detection and configure the confidence parameters for detection and tracking.
  * Finally, it will open a graphical window displaying the video feed from the device's camera, with an overlay of the detected landmarks, the currently recognized letter, and the sentence under construction.

**Instructions for Use During Execution:**

1.  Position your dominant hand in the camera's field of view, at an approximate distance of thirty to sixty centimeters, against a relatively uniform background. Adequate lighting is essential for correct landmark detection.
2.  Form the hand configuration corresponding to a letter of the Libras manual alphabet. Hold the position stable and wait as the system processes the frames. You will observe a green progress bar on the screen indicating the time remaining for letter confirmation.
3.  After two seconds of the same letter being consistently detected, it will be added to the sentence displayed at the bottom of the screen. Continue forming subsequent letters to build words or sentences.

**Keyboard Controls:**

During system execution, the following control keys are available:

  * **ESC (Escape):** Terminates the application and closes the video window.
  * **Backspace:** Removes the last letter added to the sentence, allowing for corrections.
  * **C (letter c):** Completely clears the constructed sentence, allowing you to start over.

**Exiting the System:**

To exit the system properly, press the **ESC** key. The script will release the camera resources and close all open graphical windows, returning to the terminal prompt.

### Capturing a New Dataset (Optional)

Users interested in experimenting with different datasets, expanding the system to recognize additional gestures, or collecting data specific to their own hand configurations can use the capture script to generate a custom dataset.

**Run the Capture Script:**

With the virtual environment activated, run:

```bash
python src/capture.py
```

The system will open a video window and await user instructions via the keyboard.

**Data Capture Process:**

The script allows you to capture samples organized by class (letter). To start capturing a specific letter:

1.  Press the key corresponding to the letter you want to capture (A to Z). The system will immediately begin automatically capturing landmarks whenever a hand is detected.
2.  Form the hand configuration for the chosen letter and hold it relatively stable while slightly moving your hand, subtly changing its position, rotation, and distance from the camera. This variation is important for the model to learn to recognize the letter under different conditions.
3.  The system will automatically capture up to one thousand samples for each letter, saving the data in the `data/landmarks/` directory in CSV files named after the letter (e.g., `A.csv`, `B.csv`).
4.  The screen will display the capture progress, indicating how many samples have been collected out of the one-thousand limit.

**Controls During Capture:**

  * **A-Z:** Starts or switches capture to the pressed letter.
  * **0 (zero):** Captures samples for the "none" class, representing frames where no specific letter is being formed.
  * **Spacebar:** Pauses or resumes capture for the current class.
  * **ESC:** Exits the capture script.

**Important Considerations:**

  * To obtain a robust model, it is essential to capture samples with adequate variability. Vary the lighting, camera angle, hand rotation, and distance during capture. Capture samples from different people if possible, as this increases the model's generalization ability.
  * Ensure you are correctly forming each hand configuration according to the Libras manual alphabet. Incorrect configurations during capture will result in mislabeled data, significantly harming the trained model's performance.

### Retraining the Model

After capturing a custom or modified dataset, it is necessary to train a new neural model to incorporate this data.

**Run the Training Script:**

With the virtual environment activated, run:

```bash
python src/train.py
```

The script will automatically perform the following operations:

1.  First, it will download the reference public dataset from Kaggle using the `kagglehub` library. This dataset serves as the default training base.
2.  Next, it will load all CSV files from the landmarks directory, building the feature matrices and label vectors.
3.  It will apply geometric normalization to the landmarks, making the data invariant to hand position and scale.
4.  It will execute the 5-fold stratified cross-validation process, training and evaluating the MLP model on each partition, reporting the accuracy obtained.
5.  It will calculate and display the mean accuracy and standard deviation across the partitions, providing a robust estimate of the expected performance.
6.  It will generate and save a confusion matrix detailing the classifier's patterns of correct and incorrect predictions.
7.  Finally, it will train a definitive model using all available data and save the artifacts (model, scaler, and classes) to the `models/` directory.

**Interpreting the Results:**

During training, the system will display messages indicating the progress and results of each cross-validation fold. Pay attention to the reported accuracy, which should ideally be above ninety percent for satisfactory performance in the practical application.

The saved confusion matrix can be analyzed later to identify which letters the model confuses most frequently, informing possible improvements to the dataset or model architecture.

**Training Duration:**

The time required for complete training varies depending on the dataset size and hardware processing power. On a computer with a modern processor, the typical training with the standard dataset (approximately twenty-seven thousand samples) takes between two and five minutes. Larger datasets or more modest hardware may require additional time.

-----

## Public Dataset

As part of the commitment to open science and research reproducibility, the landmark dataset used in the development and training of LibraSign has been made publicly available on the Kaggle platform. This dataset contains approximately one thousand samples for each of the twenty-seven classes (twenty-six letters plus the "none" class), totaling about twenty-seven thousand hand configuration examples.

The dataset can be accessed, viewed, and downloaded via the following link:

🔗 **[Libras Landmark Dataset (A-Z) on Kaggle](https://www.kaggle.com/datasets/heitorccf/librasign)**

**Dataset Structure:**

The dataset consists of CSV files, one for each class, where each row represents an individual sample containing sixty-three floating-point numerical values. These values correspond to the `x`, `y`, and `z` coordinates of the twenty-one landmarks extracted by MediaPipe, organized sequentially (`x₀, y₀, z₀, x₁, y₁, z₁, ..., x₂₀, y₂₀, z₂₀`).

**Using the Dataset:**

Researchers and developers interested in related work can use this dataset to:

  * Reproduce the results presented in the final year project.
  * Explore different neural network architectures and classification techniques.
  * Develop landmark-based gesture recognition systems.
  * Perform comparative performance analyses between different methodological approaches.
  * Expand the system with additional classes or complementary datasets.

When using this dataset in academic work or projects, proper citation is requested as per established academic practices.

-----

## Troubleshooting

During the installation or execution of LibraSign, some problems may be encountered depending on the operating system specifics, hardware configuration, or variations in the software environment. This section documents the most common problems and their respective solutions.

**Problem: `python` command not recognized or incorrect Python version**

  * *Symptoms:* When running `python --version`, the terminal returns an error indicating the command was not found, or it returns a Python 2.x version.
  * *Cause:* On many Unix-like systems (Linux and macOS), the `python` command points to Python 2.x for historical compatibility reasons, while Python 3.x must be invoked explicitly via the `python3` command.
  * *Solution:* In all commands presented in this guide where `python` appears, replace it with `python3`. For example, instead of `python src/predict.py`, use `python3 src/predict.py`. Alternatively, you can create an alias in your shell or modify your system's environment variables to make the `python` command point to Python 3.

**Problem: Error "Permission denied" when trying to access the camera**

  * *Symptoms:* The system starts but fails to open the camera, displaying the message "[ERROR] Could not open camera, check connection and permissions."
  * *Cause:* The operating system is blocking the application's access to the camera for privacy and security reasons.
  * *Solution on macOS:* Go to *System Preferences \> Security & Privacy \> Privacy \> Camera*, and ensure that *Terminal* (or the application you are running Python from) has permission to access the camera.
  * *Solution on Windows:* Go to *Settings \> Privacy \> Camera*, and ensure "Allow apps to access your camera" is turned on. Also, make sure "Desktop apps" have permission.
  * *Solution on Linux:* Check if your user belongs to the `video` group. Run `groups` in the terminal and check if `video` is listed. If not, add your user to the group with `sudo usermod -a -G video $USER` and restart your session.

**Problem: `ImportError` when trying to import `cv2`, `mediapipe`, or other libraries**

  * *Symptoms:* When running any script, Python returns an error similar to "ModuleNotFoundError: No module named 'cv2'" or similar for other libraries.
  * *Cause:* The dependencies were not installed correctly in the virtual environment, or the virtual environment is not activated.
  * *Solution:* First, ensure the virtual environment is activated by checking for the `(.venv)` prefix in your terminal prompt. If it is not activated, run the appropriate activation command for your system. Then, run `pip install -r requirements.txt` again to ensure all dependencies are installed. If the problem persists, try uninstalling and reinstalling the problematic library specifically, e.g.: `pip uninstall opencv-python` followed by `pip install opencv-python==4.12.0.88`.

**Problem: MediaPipe does not detect the hand or detection is unstable**

  * *Symptoms:* The system runs, but the hand landmarks are not drawn on the screen, or the detection is intermittent and unstable.
  * *Cause:* Inadequate lighting conditions, a background that is too complex or cluttered, or the hand is too close to or too far from the camera.
  * *Solution:* Improve the room's lighting, preferably using diffuse natural light or uniform artificial lighting. Position yourself against a simple and relatively uniform-colored background, avoiding complex patterns or elements that could be confused with a hand. Adjust the distance between your hand and the camera, experimenting with positions between thirty and sixty centimeters. Ensure your hand is fully visible in the frame, not being cut off by the image edges.

**Problem: Model shows low accuracy or inconsistent predictions**

  * *Symptoms:* During real-time use, predictions seem random or frequently incorrect, or during training, the reported accuracy is significantly lower than ninety percent.
  * *Cause:* The training dataset has problems, such as incorrectly labeled data, insufficient variability in the samples, or an inadequate number of examples per class.
  * *Solution:* Review the dataset capture process, ensuring the hand configurations are correct according to the Libras manual alphabet. Capture more samples for each class, ensuring variability in terms of lighting, angle, rotation, and camera distance. Consider involving multiple people in data capture to increase diversity. If using the public dataset, verify that the download was complete and the files are not corrupted.

**Problem: The training script fails to download the dataset from Kaggle**

  * *Symptoms:* During the execution of `train.py`, the system reports errors related to `kagglehub` or fails to download the dataset.
  * *Cause:* Internet connectivity issues, a firewall blocking the connection, or improper configuration of Kaggle credentials.
  * *Solution:* Check your internet connection and try again. If you are behind a corporate firewall, you may need to configure proxies. Alternatively, you can manually download the dataset from the Kaggle link, unzip it, and place the CSV files in the `data/landmarks/` directory, then modify the `train.py` script to not perform the automatic download (by commenting out the `kagglehub` download section).

**Problem: On Windows, error "cannot be loaded because running scripts is disabled on this system" when activating the virtual environment**

  * *Symptoms:* When trying to activate the virtual environment in PowerShell, an error message related to script execution policy appears.
  * *Cause:* PowerShell has a security policy that, by default, prevents the execution of unsigned scripts.
  * *Solution:* Open PowerShell as an administrator and run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`. This will allow local, unsigned scripts to run. After this setting, try activating the virtual environment again. Alternatively, you can use the Command Prompt (`cmd.exe`) instead of PowerShell, where this restriction does not apply.

**Problem: OpenCV window does not open or freezes in Linux environments without a GUI**

  * *Symptoms:* On Linux servers or environments without a graphical interface (like SSH without X forwarding), the system fails when trying to open the video window.
  * *Cause:* LibraSign was designed for environments with a full graphical interface, using OpenCV windows to display the video.
  * *Solution:* The system is not suitable for execution in GUI-less environments. For remote use, consider configuring X11 forwarding in your SSH connection (`ssh -X` or `ssh -Y`), or use remote desktop solutions like VNC. Alternatively, the code could be modified to save processed frames to disk instead of displaying them, but this is beyond the scope of the system's standard use.

**Problem: Slow performance, freezing frames, or low refresh rate**

  * *Symptoms:* The video in the system window updates very slowly, freezes frequently, or the prediction takes too long.
  * *Cause:* Insufficient hardware, other processes consuming system resources, or a low-quality camera.
  * *Solution:* Close other applications that may be consuming significant CPU or RAM. Reduce the camera's resolution if possible. On Linux systems, consider closing heavy desktop applications. Ensure your video drivers are up to date. As a last resort, consider using a computer with more robust specifications.

-----

## Applicability and Extensibility

Although LibraSign was developed specifically for recognizing the manual alphabet of Brazilian Sign Language, its modular architecture and methodology based on geometric landmarks give the system notable flexibility and potential for adaptation to various contexts.

  * **Adaptation for Other Sign Languages:** The system's structure can be readily retrained to recognize the manual alphabets of other national sign languages, such as American Sign Language (ASL), British Sign Language (BSL), or any other sign language that uses fingerspelling. The process only requires capturing a new dataset with the specific hand configurations of the target language, followed by retraining the model as described in this guide.

  * **Expansion to a Broader Vocabulary:** Researchers interested in expanding the system beyond the manual alphabet can collect samples of complete ideographic signs (words in Libras) and include them as additional classes in the dataset. This expansion would possibly demand more complex network architectures capable of modeling temporal sequences, such as Recurrent Neural Networks (RNNs) or Transformers, as many signs involve dynamic hand movement.

  * **Recognition of Custom Gestures:** The methodology can be applied to recognize custom sets of gestures in various contexts, such as gestural control of interfaces, interpretation of commands in virtual or augmented reality environments, or custom communication systems for specific needs. The versatility of MediaPipe landmarks allows
    practically any distinguishable hand configuration to be captured and classified.

  * **Integration with Other Modalities:** The current system processes only the spatial configuration of the hand. Future work could integrate facial information (expressions), body information (posture and orientation), and contextual information (position in the signing space) to move closer to a more complete recognition system for the sign language in its linguistic richness.

  * **Educational Applications:** LibraSign serves as a valuable teaching tool for concepts in machine learning, computer vision, and signal processing. Students can experiment with different network architectures, pre-processing techniques, data augmentation strategies, and validation methodologies, using the system as a practical learning platform.

-----

## Final Considerations

LibraSign represents a contribution to the field of assistive technology and the automated recognition of sign languages, demonstrating the feasibility of approaches based on geometric landmarks for classifying manual gestures. The project was developed with academic diligence, attention to reproducibility, and a commitment to disseminating knowledge through open-source code and public datasets.

It is essential to reiterate that the system, in its current state, has significant limitations that position it as a research and educational tool, not as a substitute for professional communication or sign language interpretation. Libras, like other sign languages, constitutes a complete and complex linguistic system, with its own grammar, syntax, semantics, and pragmatics that vastly transcend the mere manual spelling of letters.

The recognition of the manual alphabet, while useful in specific contexts such as spelling proper nouns or technical terms without an established sign, represents only a tiny fraction of communication in Libras. Proper understanding of the language involves facial expressions that modify grammatical meaning, movement and orientation of the hands in the three-dimensional signing space, the use of classifiers, spatial incorporation and referencing, among numerous other linguistic elements.

Therefore, it is emphasized that LibraSign should not be interpreted as a system for translating Libras in its entirety, but rather as a methodological first step toward more comprehensive systems, and as a valuable tool for the study of visual pattern recognition techniques.

Users interested in a deeper technical understanding of the system, the theoretical foundations underpinning the architectural decisions, the detailed experimental results, and discussions on related works are encouraged to consult the full academic document referenced in the next section.

-----

## References and Complementary Documentation

**📄 Complete Final Year Project Document:**

  * [HeitorFernandes-TCC\_BSI.pdf](https://github.com/Heitorccf/librasign/blob/master/HeitorFernandes-TCC_BSI.pdf)

This academic document presents, in detail, all aspects of the project, including:

  * Literature review on sign languages, assistive technologies, and gesture recognition.
  * Discussion of methodological approaches for processing visual signals.
  * Theoretical foundation on artificial neural networks and multi-layer perceptrons.
  * Detailed description of the dataset collection and preparation process.
  * Full statistical analysis of experimental results.
  * Confusion matrices and performance metrics broken down by class.
  * Discussion on system limitations and considerations for future work.
  * Reflections on the social impact of the technology and related ethical issues.

**Documentation for Libraries Used:**

For users interested in more deeply understanding the technologies employed in LibraSign, consulting the official documentation of the main libraries is recommended:

  * **MediaPipe:** [https://developers.google.com/mediapipe](https://developers.google.com/mediapipe)
  * **scikit-learn:** [https://scikit-learn.org/stable/documentation.html](https://scikit-learn.org/stable/documentation.html)
  * **OpenCV:** [https://docs.opencv.org/](https://docs.opencv.org/)
  * **NumPy:** [https://numpy.org/doc/](https://numpy.org/doc/)
  * **Pandas:** [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)

**Resources on Libras:**

For those interested in learning more about the Brazilian Sign Language and its linguistic structure:

  * **National Institute for Deaf Education (INES):** [http://www.ines.gov.br](http://www.ines.gov.br)
  * **INES Libras Dictionary:** Online resource for sign lookup.
  * **National Federation for the Education and Integration of the Deaf (FENEIS):** [https://www.feneis.org.br](https://www.feneis.org.br)

**Contributions and Feedback:**

LibraSign is an open-source project, and contributions from the community are welcome. Users who identify bugs, have suggestions for improvements, or wish to contribute code are encouraged to open issues or pull requests on the GitHub repository.

For academic questions, technical inquiries, or discussions about the project, feel free to get in touch through the channels provided in the repository.

**Acknowledgments:**

The development of LibraSign was possible thanks to the institutional support of the university, the academic guidance received, access to computational resources, and the free availability of open-source libraries by the international scientific and technological community. Special thanks to the Brazilian deaf community, whose language and culture inspire this work and motivate the development of more inclusive technologies.

**License:**

This project is distributed under an open-source license, permitting use, modification, and distribution according to the terms specified in the `LICENSE` file in the repository. When using or modifying this code, it is requested that due attribution be maintained in line with the practices of the free software community.

-----

**Last Update:** November 2025

**Author:** Heitor Fernandes
**Institution:** Bachelor in Information Systems
**Repository:** [https://github.com/heitorccf/librasign](https://github.com/heitorccf/librasign)
**Public Dataset:** [https://www.kaggle.com/datasets/heitorccf/librasign](https://www.kaggle.com/datasets/heitorccf/librasign)

-----

*This README was prepared with the goal of providing clear and accessible documentation for users of different technical experience levels. For suggestions on improving this documentation, please get in touch via the GitHub repository.*

-----

## 🇧🇷 / 🇵🇹 Versão em Português

# LibraSign

## Índice

  * [❗ Introdução](https://www.google.com/search?q=%23introducao)
  * [Fundamentação Acadêmica](https://www.google.com/search?q=%23fundamentacao-academica)
  * [Visão Geral do Sistema](https://www.google.com/search?q=%23visao-geral-do-sistema)
      * [Escopo e Limitações](https://www.google.com/search?q=%23escopo-e-limitacoes)
      * [Arquitetura do Sistema](https://www.google.com/search?q=%23arquitetura-do-sistema)
      * [Fluxo de Processamento](https://www.google.com/search?q=%23fluxo-de-processamento)
  * [Requisitos do Sistema](https://www.google.com/search?q=%23requisitos-do-sistema)
      * [Requisitos de Software](https://www.google.com/search?q=%23requisitos-de-software)
      * [Requisitos de Hardware](https://www.google.com/search?q=%23requisitos-de-hardware)
  * [Guia de Instalação](https://www.google.com/search?q=%23guia-de-instalacao)
      * [Etapa 1: Preparação do Ambiente](https://www.google.com/search?q=%23etapa-1-preparacao-do-ambiente)
      * [Etapa 2: Clonagem do Repositório](https://www.google.com/search?q=%23etapa-2-clonagem-do-repositorio)
      * [Etapa 3: Configuração do Ambiente Virtual](https://www.google.com/search?q=%23etapa-3-configuracao-do-ambiente-virtual)
      * [Etapa 4: Instalação das Dependências](https://www.google.com/search?q=%23etapa-4-instalacao-das-dependencias)
  * [Execução do Sistema](https://www.google.com/search?q=%23execucao-do-sistema)
      * [Modo de Uso Padrão](https://www.google.com/search?q=%23modo-de-uso-padrao)
      * [Captura de Novo Dataset (Opcional)](https://www.google.com/search?q=%23captura-de-novo-dataset-opcional)
      * [Retreinamento do Modelo](https://www.google.com/search?q=%23retreinamento-do-modelo)
  * [Dataset Público](https://www.google.com/search?q=%23dataset-publico)
  * [Solução de Problemas](https://www.google.com/search?q=%23solucao-de-problemas)
  * [Aplicabilidade e Extensibilidade](https://www.google.com/search?q=%23aplicabilidade-e-extensibilidade)
  * [Considerações Finais](https://www.google.com/search?q=%23consideracoes-finais)
  * [Referências e Documentação Complementar](https://www.google.com/search?q=%23referencias-e-documentacao-complementar)

-----

## ❗ Introdução

O **LibraSign** é um sistema de código aberto, desenvolvido como Trabalho de Conclusão de Curso (TCC), que utiliza técnicas de visão computacional e aprendizado de máquina para realizar o reconhecimento em tempo real de gestos correspondentes ao alfabeto manual da Língua Brasileira de Sinais (**Libras**). O projeto foi concebido para explorar metodologias de processamento de dados geométricos e classificação neural para aplicações de acessibilidade comunicacional.

É importante compreender que o LibraSign possui um escopo deliberadamente restrito: o sistema reconhece *exclusivamente* as configurações de mão correspondentes às letras do alfabeto manual (A a Z). Ele não é capaz de interpretar palavras completas, sinais compostos ou a gramática espacial da Libras. Esta delimitação foi estabelecida para permitir uma investigação acadêmica focada na eficácia de redes neurais artificiais na classificação de gestos estáticos, servindo como uma prova de conceito para futuras expansões.

O projeto destina-se primariamente ao ambiente acadêmico e educacional, constituindo uma ferramenta de estudo sobre processamento de sinais visuais e aprendizado supervisionado. Embora funcional, o sistema não foi projetado para substituir intérpretes profissionais ou para uso comunicacional cotidiano em larga escala, visto que a Libras envolve elementos linguísticos complexos que ultrapassam o escopo deste trabalho, incluindo expressões faciais, movimentos corporais e estruturas gramaticais próprias.

-----

## Fundamentação Acadêmica

Este projeto é a materialização de uma investigação científica conduzida no âmbito do curso de Bacharelado em Sistemas de Informação. A fundamentação teórica completa—incluindo revisão de literatura sobre línguas de sinais, técnicas de visão computacional, arquiteturas de redes neurais, metodologia experimental, análise estatística dos resultados e discussão sobre as implicações sociais da tecnologia assistiva—encontra-se detalhada no trabalho de conclusão de curso.

Para uma compreensão aprofundada dos fundamentos teóricos, das decisões arquiteturais, dos experimentos conduzidos e das conclusões alcançadas, recomenda-se a leitura do documento acadêmico completo, disponível neste repositório:

[cite\_start]**📄 HeitorFernandes-TCC\_BSI.pdf** [cite: 309, 310, 311, 312, 313, 314, 315]

[cite\_start]O documento acadêmico aborda tópicos essenciais como a diferenciação entre a comunicação em línguas de sinais e a datilologia (soletração manual) [cite: 505, 506][cite\_start], as limitações das abordagens baseadas em processamento de imagens brutas [cite: 623, 624, 625] [cite\_start]e a escolha por representações geométricas de *landmarks*[cite: 628, 629].

-----

## Visão Geral do Sistema

### Escopo e Limitações

Antes de utilizar o sistema, é importante que o usuário compreenda claramente o escopo de funcionalidade do LibraSign. [cite\_start]O sistema foi desenvolvido especificamente para reconhecer as configurações de mão estáticas do alfabeto manual da Libras, que correspondem às vinte e seis letras do alfabeto latino (A–Z)[cite: 412, 423]. Esta escolha metodológica foi deliberada e alinha-se com os objetivos de pesquisa do projeto.

**O que o sistema reconhece:**

  * Configurações de mão correspondentes a cada uma das letras de A a Z do alfabeto manual de Libras, quando apresentadas de forma estática e isolada diante da câmera.

**O que o sistema NÃO reconhece:**

  * Palavras completas em Libras, que frequentemente são representadas por sinais únicos e não pela soletração letra a letra.
  * Sinais compostos ou ideográficos que constituem o vocabulário padrão da língua.
  * [cite\_start]Expressões faciais, movimento corporal ou utilização do espaço de sinalização, elementos essenciais da gramática de Libras[cite: 483, 490, 491, 492, 493].
  * Variações regionais ou dialetos da língua de sinais.
  * [cite\_start]Transições dinâmicas entre letras ou gestos em movimento contínuo[cite: 509, 511, 512].

Esta delimitação posiciona o LibraSign como uma ferramenta educacional e de pesquisa, adequada para o estudo de técnicas de reconhecimento de padrões e para aplicações didáticas de ensino do alfabeto manual, mas não como um tradutor completo da língua de sinais. O projeto estabelece uma base que pode ser expandida em trabalhos futuros para incluir vocabulário mais amplo e elementos linguísticos adicionais.

### Arquitetura do Sistema

O LibraSign foi arquitetado seguindo uma metodologia modular que separa claramente as responsabilidades de cada componente do sistema. Esta organização facilita a manutenção, o teste e a eventual expansão das funcionalidades. A arquitetura compreende três módulos principais:

  * **Módulo de Captura de Dados (`src/capture.py`):** Este componente é responsável pela aquisição de dados de treinamento. [cite\_start]Utilizando a biblioteca **MediaPipe** desenvolvida pelo Google, o módulo acessa a câmera do dispositivo e realiza a detecção em tempo real das mãos presentes no campo de visão[cite: 342, 641, 642]. [cite\_start]Para cada *frame* capturado, o MediaPipe identifica 21 pontos de referência anatômicos (*landmarks*) na mão detectada, extraindo suas coordenadas tridimensionais (x, y, z) no espaço normalizado[cite: 629, 642, 646]. [cite\_start]Estes dados geométricos, ao invés de imagens brutas em pixels, são persistidos em arquivos CSV organizados por classe, criando um *dataset* leve e estruturado que facilita o processamento posterior[cite: 656, 663].

  * **Módulo de Treinamento (`src/train.py`):** Este componente implementa o *pipeline* completo de aprendizado supervisionado. Inicialmente, o módulo carrega o *dataset* de *landmarks* a partir dos arquivos CSV. Em seguida, aplica uma transformação de normalização geométrica que torna os dados invariantes à posição absoluta da mão no quadro e à escala (distância da câmera), centralizando os pontos em relação ao pulso e normalizando pelo comprimento característico da mão. [cite\_start]Após a normalização, os dados são padronizados utilizando o `StandardScaler` para apresentarem média zero e variância unitária[cite: 695, 696, 697, 699]. [cite\_start]O modelo escolhido é um **Perceptron Multicamadas (MLP)** com duas camadas ocultas (128 e 64 neurônios)[cite: 704, 726], treinado através do algoritmo de retropropagação. A avaliação do desempenho é conduzida através de validação cruzada estratificada (5 folds), garantindo estimativas robustas da capacidade de generalização. [cite\_start]Ao final, o modelo treinado, o objeto de padronização (`scaler`) e o mapeamento de classes são serializados para uso na inferência[cite: 740].

  * **Módulo de Predição em Tempo Real (`src/predict.py`):** Este é o módulo de interface com o usuário, responsável pela aplicação prática do modelo treinado. [cite\_start]O componente carrega os artefatos persistidos (modelo, `scaler` e classes)[cite: 746], inicializa a captura de vídeo e processa cada *frame* em tempo real. [cite\_start]Para cada detecção de mão, as coordenadas dos *landmarks* são extraídas, normalizadas e padronizadas exatamente da mesma forma que durante o treinamento, garantindo a consistência dos dados de entrada[cite: 750, 751, 753]. O vetor resultante é submetido ao classificador neural, que retorna probabilidades para cada classe. O sistema implementa duas estratégias de estabilização: um filtro de votação majoritária sobre os últimos 10 *frames* para suavizar predições ruidosas, e um mecanismo de confirmação temporal que exige que uma letra permaneça estável por 2 segundos antes de ser adicionada à frase em construção. O resultado é apresentado visualmente na tela, juntamente com indicadores de confiança e a sentença formada.

### Fluxo de Processamento

Para auxiliar na compreensão da operação do sistema, apresenta-se a seguir o fluxo sequencial de processamento desde a captura do gesto até a apresentação do resultado:

1.  [cite\_start]**Aquisição do *Frame*:** O sistema captura continuamente *frames* da câmera do dispositivo em tempo real[cite: 651].
2.  [cite\_start]**Detecção da Mão:** Cada *frame* é processado pelo modelo de detecção de mãos do MediaPipe, que identifica a presença e localização de mãos na imagem[cite: 653].
3.  [cite\_start]**Extração de *Landmarks*:** O MediaPipe identifica 21 pontos anatômicos na mão detectada (articulações, pontas dos dedos, pulso)[cite: 642, 647]. [cite\_start]Cada *landmark* é representado por suas coordenadas tridimensionais (x, y, z)[cite: 646, 655].
4.  **Normalização Geométrica:** Os *landmarks* brutos são transformados para garantir invariância. Primeiro, todos os pontos são transladados para que o pulso (*landmark* zero) fique na origem. [cite\_start]Em seguida, os pontos são escalonados pela distância entre o pulso e a base do dedo médio (*landmark* nove), tornando a representação independente do tamanho e posição absoluta da mão[cite: 975].
5.  [cite\_start]**Padronização Estatística:** Os dados normalizados geometricamente são padronizados utilizando o `StandardScaler` treinado (subtraindo a média e dividindo pelo desvio padrão de cada característica)[cite: 699, 751].
6.  [cite\_start]**Classificação Neural:** O vetor de características padronizado (63 dimensões) [cite: 655] [cite\_start]é propagado através das camadas do MLP (128x64 neurônios)[cite: 726]. [cite\_start]A camada de saída produz probabilidades para cada classe (letra)[cite: 756, 757, 758].
7.  **Estabilização Temporal:** Para reduzir oscilações, o sistema aplica um filtro de votação majoritária sobre as últimas 10 predições. Além disso, uma letra só é confirmada se permanecer como predição predominante por 2 segundos consecutivos.
8.  [cite\_start]**Apresentação dos Resultados:** O sistema renderiza sobre o vídeo os *landmarks* detectados, a letra reconhecida, uma barra de progresso para confirmação e, na parte inferior, a sentença formada pelas letras confirmadas[cite: 759, 833, 834].

-----

## Requisitos do Sistema

### Requisitos de Software

Para a correta execução do LibraSign, é necessário que o ambiente de desenvolvimento atenda aos seguintes requisitos de software:

  * **Sistema Operacional:** O sistema foi desenvolvido e testado em ambientes Linux (distribuições baseadas em Debian e Fedora), macOS (versões 11 e superiores) e Windows 10/11.
  * **Interpretador Python:** É necessária a instalação do Python na versão **3.11.13**, conforme especificado no desenvolvimento do projeto. Versões anteriores à 3.9 não são suportadas. Versões posteriores podem funcionar, mas não foram testadas.
  * **Gerenciador de Pacotes pip:** A instalação das dependências do projeto é realizada através do `pip`.
  * **Câmera Funcional:** O sistema requer acesso a uma câmera (webcam integrada ou externa) para captura de vídeo em tempo real.
  * **Bibliotecas Python Essenciais:** As seguintes bibliotecas e suas versões são importantes para o funcionamento adequado:

<!-- end list -->

```
# Versão do Python recomendada para este projeto: 3.11.13

scikit-learn==1.7.2
numpy==2.2.6
pandas==2.3.2
opencv-python==4.12.0.88
mediapipe==0.10.14
kagglehub==0.3.13
```

### Requisitos de Hardware

Embora o LibraSign tenha sido otimizado para execução em hardware modesto, certos requisitos mínimos devem ser atendidos:

  * **Processador:** Recomenda-se um processador com pelo menos dois núcleos físicos (ex: Intel Core i3 8ª ger, AMD Ryzen 3 ou equivalentes).
  * **Memória RAM:** Um mínimo de 4 GB de RAM é necessário. Recomenda-se 8 GB ou mais para operação confortável, especialmente durante o treinamento.
  * **Armazenamento:** O projeto e o *dataset* público ocupam menos de 100 MB. Recomenda-se ter pelo menos 1 GB de espaço livre.
  * **Câmera:** Câmera com resolução mínima de 640x480 pixels (VGA) e taxa de captura de pelo menos 15 frames por segundo (FPS). Câmeras HD (720p) ou superiores com 30 FPS proporcionam melhor experiência.
  * **Iluminação:** Condições adequadas de iluminação são cruciais. [cite\_start]Recomenda-se ambiente bem iluminado, evitando contraluz intenso ou sombras fortes que possam dificultar a detecção dos *landmarks*[cite: 554, 564, 676].
  * **Sistema Gráfico:** Suporte básico para exibição de janelas gráficas (OpenCV). O sistema não funcionará em ambientes de servidor sem interface gráfica (GUI).

-----

## Guia de Instalação

Este guia conduzirá o usuário através das etapas para preparar o ambiente de desenvolvimento e instalar o LibraSign.

### Etapa 1: Preparação do Ambiente

Garanta que o interpretador Python (versão 3.11.x) esteja instalado e configurado corretamente no sistema.

**Verificação da Instalação do Python:**

Abra seu terminal (Linux/macOS) ou Prompt de Comando/PowerShell (Windows) e execute:

```bash
python --version
```

Em alguns sistemas (Linux/macOS), pode ser necessário usar `python3`:

```bash
python3 --version
```

O comando deve retornar uma versão 3.11.x (idealmente `Python 3.11.13`). Se o comando não for reconhecido ou a versão for inferior, instale ou atualize o Python.

**Instalação do Python (se necessário):**

  * **Linux (Debian/Ubuntu):**
    ```bash
    sudo apt update
    sudo apt install python3.11 python3.11-venv python3-pip
    ```
  * **macOS:** Recomenda-se utilizar o Homebrew:
    ```bash
    brew install python@3.11
    ```
  * **Windows:** Baixe o instalador oficial do `python.org`, marcando a opção "**Add Python to PATH**" durante a instalação.

**Verificação do pip:**

Verifique se o `pip` está disponível:

```bash
python3 -m pip --version
```

### Etapa 2: Clonagem do Repositório

Com o Python instalado, obtenha uma cópia local do repositório. Certifique-se de ter o **Git** instalado (`git --version`).

**Clonagem do Repositório:**

Navegue até o diretório onde deseja armazenar o projeto e clone o repositório:

```bash
git clone https://github.com/heitorccf/librasign.git
```

**Navegação até o Diretório do Projeto:**

Entre no diretório recém-criado:

```bash
cd librasign
```

Todos os comandos subsequentes devem ser executados a partir desta pasta.

### Etapa 3: Configuração do Ambiente Virtual

O uso de um ambiente virtual (venv) é recomendado para isolar as dependências do projeto.

**Criação do Ambiente Virtual:**

  * **Linux e macOS:**
    ```bash
    python3 -m venv .venv
    ```
  * **Windows:**
    ```bash
    python -m venv .venv
    ```

**Ativação do Ambiente Virtual:**

  * **Linux e macOS:**
    ```bash
    source .venv/bin/activate
    ```
  * **Windows (Prompt de Comando):**
    ```bash
    .venv\Scripts\activate.bat
    ```
  * **Windows (PowerShell):**
    ```bash
    .venv\Scripts\Activate.ps1
    ```
    *(Nota: No PowerShell, talvez seja necessário ajustar a política de execução com `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)*

Após a ativação, seu terminal deve exibir `(.venv)` no início do *prompt*.

### Etapa 4: Instalação das Dependências

Com o ambiente virtual ativado, instale as bibliotecas listadas no `requirements.txt`.

**Atualização do pip (recomendado):**

```bash
python -m pip install --upgrade pip
```

**Instalação das Dependências:**

```bash
pip install -r requirements.txt
```

O `pip` instalará todas as bibliotecas necessárias, incluindo `opencv-python`, `mediapipe`, `scikit-learn` e `kagglehub`.

**Verificação da Instalação:**

Teste se as bibliotecas principais foram instaladas:

```bash
python -c "import cv2, mediapipe, sklearn, numpy, pandas; print('Todas as bibliotecas foram importadas com sucesso')"
```

Se a mensagem de sucesso for exibida, o ambiente está pronto.

-----

## Execução do Sistema

O sistema oferece diferentes modos de operação.

### Modo de Uso Padrão

Este modo utiliza o modelo pré-treinado (disponível na pasta `models/`) para reconhecimento em tempo real.

**Execução do Tradutor em Tempo Real:**

Com o ambiente virtual ativado, execute o script de predição:

```bash
python src/predict.py
```

[cite\_start]O sistema carregará o modelo [cite: 746] [cite\_start]e abrirá uma janela gráfica exibindo o vídeo da sua câmera[cite: 747].

**Instruções de Uso Durante a Execução:**

1.  Posicione sua mão no campo de visão da câmera, contra um fundo relativamente uniforme.
2.  Forme a configuração de mão correspondente a uma letra do alfabeto manual. Mantenha a posição estável.
3.  [cite\_start]Uma barra de progresso verde indicará o tempo para confirmação do gesto[cite: 811].
4.  [cite\_start]Após 2 segundos, a letra será confirmada e adicionada à frase na parte inferior da tela[cite: 821, 833, 834].

**Controles do Teclado:**

  * **ESC (Escape):** Encerra a aplicação.
  * **Backspace:** Remove a última letra adicionada à frase.
  * **C (letra cê):** Limpa completamente a frase.

### Captura de Novo Dataset (Opcional)

Usuários que desejam capturar seus próprios dados (para expandir o *dataset* ou melhorar a precisão para sua própria mão) podem usar o script de captura.

**Execução do Script de Captura:**

```bash
python src/capture.py
```

O sistema abrirá uma janela de vídeo e aguardará seus comandos.

**Processo de Captura de Dados:**

1.  Pressione a tecla da letra que deseja capturar (A a Z). A captura iniciará automaticamente quando uma mão for detectada.
2.  Forme o gesto da letra escolhida. [cite\_start]Mova levemente a mão (posição, rotação, distância) para criar variabilidade nos dados[cite: 661].
3.  O sistema capturará até 1000 amostras por letra, salvando os *landmarks* no diretório `data/landmarks/` em arquivos CSV (ex: `A.csv`).
4.  A tela exibirá o progresso da captura (ex: `GRAVANDO 'A' (123/1000)`).

**Controles Durante a Captura:**

  * **A-Z:** Inicia ou alterna a captura para a letra pressionada.
  * **0 (zero):** Captura amostras da classe "nenhum" (mão relaxada ou ausente).
  * **Espaço:** Pausa ou retoma a captura para a classe atual.
  * **ESC:** Encerra o script de captura.

### Retreinamento do Modelo

Após capturar um *dataset* personalizado, é necessário treinar um novo modelo para incorporar esses dados.

**Execução do Script de Treinamento:**

```bash
python src/train.py
```

O script realizará as seguintes operações:

1.  Baixará o *dataset* público de referência do Kaggle usando `kagglehub` (para garantir uma base).
2.  Carregará todos os arquivos CSV do diretório de *landmarks* (incluindo os que você capturou).
3.  Aplicará a normalização geométrica e a padronização (`StandardScaler`).
4.  [cite\_start]Executará a validação cruzada estratificada (5 folds) para avaliar o modelo, reportando a acurácia de cada *fold*[cite: 791].
5.  Exibirá a acurácia média e o desvio padrão.
6.  Treinará um modelo final usando todos os dados.
7.  Salvará os novos artefatos (`librasign_mlp.pkl`, `scaler.pkl`, `classes.npy`) no diretório `models/`, substituindo os antigos.

[cite\_start]O tempo de treinamento com o *dataset* padrão (aprox. 27.000 amostras) é de poucos minutos em um computador moderno[cite: 788].

-----

## Dataset Público

O *dataset* de *landmarks* usado no desenvolvimento do LibraSign foi disponibilizado publicamente na plataforma Kaggle. Ele contém aproximadamente 1.000 amostras para cada uma das 27 classes (26 letras + "nenhum"), totalizando cerca de 27.000 exemplos.

O *dataset* pode ser acessado, visualizado e baixado através do seguinte link:

🔗 **[Libras Landmark Dataset (A-Z) no Kaggle](https://www.kaggle.com/datasets/heitorccf/librasign)**

**Estrutura do Dataset:**

[cite\_start]O *dataset* consiste em arquivos CSV (um para cada classe), onde cada linha representa uma amostra contendo 63 valores numéricos (coordenadas x, y, z dos 21 *landmarks* da mão)[cite: 655, 663].

**Utilização do Dataset:**

Pesquisadores e desenvolvedores podem utilizar este *dataset* para:

  * Reproduzir os resultados apresentados no TCC.
  * Explorar diferentes arquiteturas de redes neurais e técnicas de classificação.
  * Desenvolver outros sistemas de reconhecimento de gestos baseados em *landmarks*.
  * Expandir o sistema com classes adicionais.

-----

## Solução de Problemas

Esta seção documenta problemas comuns e suas soluções.

**Problema: Comando `python` não reconhecido ou versão incorreta**

  * *Sintomas:* O terminal retorna "comando não encontrado" ou exibe uma versão 2.x do Python.
  * *Causa:* Em sistemas Unix (Linux/macOS), `python` pode apontar para o Python 2.x. O Python 3.x deve ser invocado com `python3`.
  * *Solução:* Use `python3` em vez de `python` para todos os comandos (ex: `python3 src/predict.py`).

**Problema: Erro "Permission denied" ou "[ERRO] Não foi possível abrir a câmera"**

  * *Sintomas:* O sistema falha ao acessar a webcam.
  * *Causa:* O sistema operacional está bloqueando o acesso do script à câmera por motivos de privacidade.
  * *Solução (macOS):* Vá em *Preferências do Sistema \> Segurança e Privacidade \> Privacidade \> Câmera* e autorize o *Terminal* (ou seu editor de código).
  * *Solução (Windows):* Vá em *Configurações \> Privacidade \> Câmera* e ative "Permitir que aplicativos acessem sua câmera" e "Permitir que aplicativos da área de trabalho acessem a câmera".
  * *Solução (Linux):* Verifique se seu usuário pertence ao grupo `video` (com o comando `groups`). Se não pertencer, adicione-o com `sudo usermod -a -G video $USER` e reinicie a sessão.

**Problema: `ImportError` ou "ModuleNotFoundError: No module named 'cv2'"**

  * *Sintomas:* O Python não encontra bibliotecas como `cv2` (OpenCV) ou `mediapipe`.
  * *Causa:* As dependências não foram instaladas, ou o ambiente virtual não está ativado.
  * *Solução:* Verifique se o `(.venv)` aparece no seu terminal. Se não, ative o ambiente virtual. Em seguida, execute `pip install -r requirements.txt` novamente.

**Problema: MediaPipe não detecta a mão ou a detecção é instável**

  * *Sintomas:* Os *landmarks* da mão não aparecem ou piscam na tela.
  * *Causa:* Condições de iluminação inadequadas, fundo muito complexo ou a mão está muito perto/longe da câmera.
  * *Solução:* Melhore a iluminação do ambiente. Use um fundo simples e de cor uniforme. Ajuste a distância da mão para a câmera (entre 30 e 60 cm). Certifique-se de que a mão inteira está visível.

**Problema: Modelo com baixa acurácia ou predições inconsistentes**

  * *Sintomas:* O sistema confunde letras frequentemente, ou a acurácia no retreinamento é baixa.
  * [cite\_start]*Causa:* *Dataset* de treinamento com problemas (rótulos errados, pouca variabilidade)[cite: 661]. [cite\_start]A confusão entre pares semelhantes (como 'M'/'N', 'G'/'Q', 'F'/'T') é uma limitação conhecida, pois eles são geometricamente muito próximos[cite: 977, 980, 981, 983, 984].
  * *Solução:* Para problemas gerais de acurácia, revise seu *dataset* personalizado, garantindo que os gestos estão corretos e capture mais amostras com variabilidade. [cite\_start]A confusão entre pares semelhantes é uma limitação atual do modelo[cite: 976, 986, 987].

**Problema: Script de treinamento falha ao baixar o *dataset* do Kaggle**

  * *Sintomas:* Erros relacionados ao `kagglehub` durante a execução de `train.py`.
  * *Causa:* Problemas de conexão com a internet ou firewall.
  * *Solução:* Verifique sua conexão. Você também pode baixar o *dataset* manualmente pelo link do Kaggle, descompactar os arquivos CSV na pasta `data/landmarks/` e, em seguida, comentar ou remover as linhas referentes ao `kagglehub.dataset_download` no script `src/train.py`.

**Problema: Janela do OpenCV não abre (ambientes Linux sem GUI)**

  * *Sintomas:* O script falha ao tentar abrir a janela de vídeo em servidores ou SSH.
  * *Causa:* O sistema requer uma interface gráfica (GUI) para exibir o vídeo do OpenCV.
  * *Solução:* O sistema não é feito para ambientes sem GUI. Use-o em um desktop ou configure *X11 forwarding* na sua conexão SSH (se aplicável).

-----

## Aplicabilidade e Extensibilidade

Embora o LibraSign seja focado no alfabeto manual da Libras, sua arquitetura baseada em *landmarks* geométricos oferece flexibilidade para adaptação:

  * **Adaptação para Outras Línguas de Sinais:** A estrutura pode ser retreinada para reconhecer alfabetos manuais de outras línguas (como ASL, BSL, etc.). O processo requer apenas a captura de um novo *dataset* com as configurações de mão da língua-alvo e o retreinamento do modelo.
  * **Expansão para Vocabulário Mais Amplo:** O sistema pode ser expandido para reconhecer sinais ideográficos (palavras completas). [cite\_start]Isso exigiria a coleta de amostras desses sinais e, possivelmente, a mudança para arquiteturas de rede capazes de modelar sequências temporais (como RNNs ou Transformers), já que muitos sinais envolvem movimento dinâmico[cite: 590, 593, 594, 595].
  * **Reconhecimento de Gestos Personalizados:** A metodologia pode ser aplicada para reconhecer conjuntos de gestos personalizados para outros fins, como controle de interfaces, comandos em realidade virtual ou sistemas de comunicação customizados.
  * **Aplicações Educacionais:** O LibraSign serve como uma ferramenta didática para o ensino de conceitos de aprendizado de máquina, visão computacional e processamento de sinais.

-----

## Considerações Finais

O LibraSign demonstra a viabilidade de abordagens baseadas em *landmarks* geométricos para a classificação de gestos manuais. [cite\_start]O projeto foi desenvolvido com atenção à reprodutibilidade, disponibilizando o código e o *dataset* publicamente[cite: 910, 912, 914, 915].

[cite\_start]É essencial reiterar que o sistema, em seu estado atual, possui limitações que o posicionam como ferramenta de pesquisa e educação, não como substituto para interpretação profissional[cite: 505]. [cite\_start]A Libras é um sistema linguístico completo que transcende a soletração manual [cite: 471, 474][cite\_start], envolvendo gramática espacial, expressões faciais e movimento corporal[cite: 483, 490].

O reconhecimento do alfabeto manual representa apenas uma pequena fração da comunicação em Libras. Portanto, o LibraSign deve ser visto como um primeiro passo metodológico em direção a sistemas mais completos, e como uma ferramenta para o estudo de técnicas de reconhecimento de padrões visuais.

[cite\_start]Usuários interessados no entendimento técnico do sistema, nos fundamentos teóricos, nos resultados experimentais e nas discussões sobre trabalhos relacionados são encorajados a consultar o documento acadêmico completo referenciado na próxima seção[cite: 317, 318, 319, 340].

-----

## Referências e Documentação Complementar

**📄 Trabalho de Conclusão de Curso Completo:**

  * [HeitorFernandes-TCC\_BSI.pdf](https://github.com/Heitorccf/librasign/blob/master/HeitorFernandes-TCC_BSI.pdf)

Este documento acadêmico apresenta de forma detalhada todos os aspectos do projeto, incluindo:

  * [cite\_start]Revisão bibliográfica sobre línguas de sinais, tecnologias assistivas e reconhecimento de gestos[cite: 430, 431].
  * [cite\_start]Discussão sobre abordagens metodológicas para processamento de sinais visuais[cite: 523].
  * [cite\_start]Fundamentação teórica sobre redes neurais (MLP)[cite: 344, 702, 703, 704, 705].
  * [cite\_start]Descrição do processo de coleta e preparação do *dataset*[cite: 341, 343, 636, 637].
  * [cite\_start]Análise estatística dos resultados experimentais[cite: 762, 763, 764, 765].
  * [cite\_start]Discussão sobre limitações do sistema e trabalhos futuros[cite: 923, 968, 1003, 1004, 1005].

**Documentação das Bibliotecas Utilizadas:**

Para usuários interessados em compreender mais profundamente as tecnologias empregadas, recomenda-se a consulta da documentação oficial:

  * [cite\_start]**MediaPipe:** [https://developers.google.com/mediapipe](https://developers.google.com/mediapipe) [cite: 1034]
  * [cite\_start]**scikit-learn:** [https://scikit-learn.org/stable/documentation.html](https://scikit-learn.org/stable/documentation.html) [cite: 1048, 1049, 1052, 1053, 1054]
  * **OpenCV:** [https://docs.opencv.org/](https://docs.opencv.org/)
  * **NumPy:** [https://numpy.org/doc/](https://numpy.org/doc/)
  * **Pandas:** [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)

**Recursos sobre Libras:**

Para aqueles interessados em aprender mais sobre a Língua Brasileira de Sinais:

  * **Instituto Nacional de Educação de Surdos (INES):** [http://www.ines.gov.br](http://www.ines.gov.br)
  * **Dicionário de Libras do INES:** Recurso online para consulta de sinais.
  * **Federação Nacional de Educação e Integração dos Surdos (FENEIS):** [https://www.feneis.org.br](https://www.feneis.org.br)

**Contribuições e Feedback:**

O LibraSign é um projeto de código aberto e contribuições da comunidade são bem-vindas. Usuários que identificarem *bugs*, tiverem sugestões de melhorias ou desejarem contribuir com código são encorajados a abrir *issues* ou *pull requests* no repositório do GitHub.

**Agradecimentos:**

[cite\_start]O desenvolvimento do LibraSign foi possível graças ao apoio institucional, à orientação acadêmica [cite: 330] [cite\_start]e à disponibilização gratuita de bibliotecas de código aberto pela comunidade[cite: 337, 338]. Agradecimentos especiais à comunidade surda brasileira, cuja língua e cultura inspiram este trabalho.

**Licença:**

Este projeto é distribuído sob a licença GNU General Public License v3.0, permitindo uso, modificação e distribuição de acordo com os termos especificados no arquivo `LICENSE`.

-----

**Última Atualização:** Novembro de 2025

[cite\_start]**Autor:** Heitor Fernandes [cite: 312, 316]
[cite\_start]**Instituição:** Bacharelado em Sistemas de Informação [cite: 319]
[cite\_start]**Repositório:** [https://github.com/heitorccf/librasign](https://github.com/heitorccf/librasign) [cite: 614, 1001]
**Dataset Público:** [https://www.kaggle.com/datasets/heitorccf/librasign](https://www.kaggle.com/datasets/heitorccf/librasign)

-----

*Este README foi elaborado com o objetivo de fornecer uma documentação clara e acessível para usuários de diferentes níveis de experiência técnica. Para sugestões de melhorias nesta documentação, por favor, entre em contato através do repositório do GitHub.*