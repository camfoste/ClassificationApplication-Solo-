# SnapNLearn
SnapNLearn is an application that accepts and processes a CIFAR-100 generated image from the user's disc/storage on their computer/device and classifies that image with a Super and Sub class label defined by the CIFAR-100 dataset.

✨ <b>Project Information: </b> ✨

  1. Uses 'generate_cifar_images' to create example images from CIFAR-100
  2. Custom CNN architecture with convolutional layers, pooling, and fully connected layers
  3. Takes the provided CNN model and creates a fully trained model with loss and accuracy tracking
  4. Tests the model and generates metrics such as accuracy, precision, recall, and F1-score
  5. Utilizes a GUI for user convenience in loading and classifying a photo

<b>Requirements: </b>

  1. Python: https://www.python.org/downloads/
  2. WinRAR: https://www.win-rar.com/download.html?&L=5&subD=true
     (Or other extractors/archivers such as 7-Zip or PeaZip)
  3. CIFAR-100 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
  4. Package manager 'pip': in CMD `python get-pip.py`
  5. Python Packages: in CMD
     ```
     pip install torch torchvision
     pip install tensorboard
     pip install DateTime
     pip install numpy
     pip install matplotlib
     pip install scikit-learn
     pip install tqdm
     ```

<b>Execution Instructions: </b>

  1. Install/download all requirements
  2. Download model_cnn.py, train.py, test.py, and GUI.py from the repository <br />
  ❗ Ensure they are all in the same folder ❗
  3. Within the same folder, create a folder labeled 'data' and extract the contents of 'cifar-100-python.tar.gz' into it <br />
  ❗ You cannot extract a tar file without an extractor program ❗
  4. Run train.py in the command prompt to generate the trained model
  ```
  python train.py
  ```
   • Trains a model from scratch using 80% of the CIFAR-100 set for training and 20% for validation <br />
   • Logs training and validation metrics to TensorBoard in the 'runs' folder <br />
   • Saves trained model as a unique path with 'model_YYYYMMDD_HHMMSS.pth' <br />
   
  5. View loss graphs with TensorBoard, type in the command prompt
  ```
  tensorboard --logdir=runs
  ```
   • You will be given a URL to copy and paste into your web browser <br />
   
  6. Run test.py in the command prompt to run evaluations
  ```
  python test.py
  ```
   • Loads the most recent saved model <br />
   • 📊 Evaluates: <br />
   &nbsp;&nbsp;&nbsp;&nbsp;• Accuracy per class and superclass <br />
   &nbsp;&nbsp;&nbsp;&nbsp;• Precision <br />
   &nbsp;&nbsp;&nbsp;&nbsp;• Recall <br />
   &nbsp;&nbsp;&nbsp;&nbsp;• F1-score <br />
   • Results are saved to a '.txt' file via create_file()

  7. Run the GUI for user image classification
  ```
  python GUI.py
  ```
   • Select an image file (.jpg, .jpeg, or .png) with the 'Select Image' button <br />
   • Classify the loaded image using the most recently saved model with the 'Classify' button <br />
   • Displays predicted fine class and its corresponding superclass
