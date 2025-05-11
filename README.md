SnapNLearn is an application that accepts and processes a CIFAR-100 generated image from the user's disc/storage on their computer/device and classifies that image with a Super and Sub class label defined by the CIFAR-100 dataset.

✨ Project Information: ✨

Uses 'generate_cifar_images' to create example images from CIFAR-100
Custom CNN architecture with convolutional layers, pooling, and fully connected layers
Takes the provided CNN model and creates a fully trained model with loss and accuracy tracking
Tests the model and generates metrics such as accuracy, precision, recall, and F1-score
Utilizes a GUI for user convenience in loading and classifying a photo
Requirements:

Python: https://www.python.org/downloads/
WinRAR: https://www.win-rar.com/download.html?&L=5&subD=true (Or other extractors/archivers such as 7-Zip or PeaZip)
CIFAR-100 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
Package manager 'pip': in CMD python get-pip.py
Python Packages: in CMD
pip install torch torchvision
pip install tensorboard
pip install DateTime
pip install numpy
pip install matplotlib
pip install scikit-learn
pip install tqdm
Execution Instructions:

Install/download all requirements
Download model_cnn.py, train.py, test.py, and GUI.py from the repository
❗ Ensure they are all in the same folder ❗
Within the same folder, create a folder labeled 'data' and extract the contents of 'cifar-100-python.tar.gz' into it
❗ You cannot extract a tar file without an extractor program ❗
Run train.py in the command prompt to generate the trained model
python train.py
• Trains a model from scratch using 80% of the CIFAR-100 set for training and 20% for validation
• Logs training and validation metrics to TensorBoard in the 'runs' folder
• Saves trained model as a unique path with 'model_YYYYMMDD_HHMMSS.pth'

View loss graphs with TensorBoard, type in the command prompt
tensorboard --logdir=runs
• You will be given a URL to copy and paste into your web browser

Run test.py in the command prompt to run evaluations
python test.py
• Loads the most recent saved model
• 📊 Evaluates:
    • Accuracy per class and superclass
    • Precision
    • Recall
    • F1-score
• Results are saved to a '.txt' file via create_file()

Run the GUI for user image classification
python GUI.py
• Select an image file (.jpg, .jpeg, or .png) with the 'Select Image' button
• Classify the loaded image using the most recently saved model with the 'Classify' button
• Displays predicted fine class and its corresponding superclass
