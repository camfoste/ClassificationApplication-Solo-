'''
Taken from https://www.kaggle.com/code/vbookshelf/cifar-100-how-to-create-csv-files-and-jpg-images
'''

import pandas as pd
import numpy as np
import os
import sys
import logging
import cv2
import shutil

#from tqdm import tqdm
# tqdm doesn't work well in colab.
# This is the solution:
# https://stackoverflow.com/questions/41707229/tqdm-printing-to-newline
import tqdm
#for i in tq.tqdm(...):

# === Logging Setup ===
def setup_logging():
    # Initializes logging to display timestamps and log levels
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import matplotlib.pyplot as plt
SPLIT = sys.argv[1]

def unpickle(file):
    # Loads a pickle file and returns the dictionary
    import pickle
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
        
    return data

# === Create Image Output Directory ===
def create_image_directory(split):
    # Creates a folder named `train_images` or `test_images` to save the generated images
    image_dir = f"{split}_images"
    if not os.path.isdir(image_dir):
        os.mkdir(image_dir)
    return image_dir
    
path = './data/cifar-100-python/' + SPLIT

train_dict = unpickle(path)

train_dict.keys()

path = './data/cifar-100-python/meta'

names_dict = unpickle(path)

names_dict.keys()

matrix = train_dict[b'data']

fine_labels_list = train_dict[b'fine_labels']
coarse_labels_list = train_dict[b'coarse_labels']

fine_label_names_list = names_dict[b'fine_label_names']
coarse_label_names_list = names_dict[b'coarse_label_names']
df_train = pd.DataFrame(fine_labels_list, columns=['fine_labels'])

# Create new columns
df_train['coarse_labels'] = coarse_labels_list
df_train['image_num'] = df_train.index + 100000
if os.path.isdir(SPLIT + '_images') == False:
    train_images = SPLIT + '_images'
    print(train_images)
    os.mkdir(train_images)

for i in range(0, matrix.shape[0]):
    
    # Get the image_id from the df_train dataframe
    image_id = df_train.loc[i, 'image_num']


    # Select an image
    row = matrix[i]

    # Extract each channel
    ch0 = row[0:1024] 
    ch1 = row[1024:2048]
    ch2 = row[2048:]

    # === Convert Row Data to Image and Save ===
    def save_image(image_data, image_id, image_dir):
        
    # Reshape to 32x32
    ch0 = np.reshape(ch0, (32,32)) # red
    ch1 = np.reshape(ch1, (32,32)) # green
    ch2 = np.reshape(ch2, (32,32)) # blue

    # Stack the matrices along the channel axis
    image = np.dstack((ch0, ch1, ch2))

    
    # Save the image in the folder
    # that we created.
    fname = str(image_id) + '.png'
    dst = os.path.join(SPLIT + '_images', fname)

    # === Load Dataset ===
    data_file = os.path.join(DATA_DIR, split)  # path to train or test data file
    train_dict = unpickle(data_file)  # Contains image data and labels
    names_dict = unpickle(META_FILE)  # Contains label name info
    
    # If cv2.COLOR_RGB2BGR is not used then the saved images appear blue.
    im_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(dst, im_bgr)

# === Run Script ===
if __name__ == "__main__":
    setup_logging()  # Set up logging format
    main()  # Run the conversion script
