import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split

import pandas as pd
import numpy as np
import os
import cv2
import pickle

def load_cifar100(batch_size=32, val_split=0.2):
    """
    Load CIFAR-100 dataset and create DataLoaders for training, validation, and test sets.
    
    Parameters:
    batch_size (int): Batch size for DataLoaders.
    val_split (float): Fraction of training set to use as validation set.

    Returns:
    tuple: DataLoaders for training, validation, and test sets.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    trainset, valset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def unpickle(file):
    """
    Unpickle a file and return the dictionary.
    
    Parameters:
    file (str): Path to the pickle file.

    Returns:
    dict: Unpickled data.
    """
    try:
        with open(file, 'rb') as fo:
            data_dict = pickle.load(fo, encoding='bytes')
        return data_dict
    except (FileNotFoundError, pickle.PickleError) as e:
        print(f"Error loading {file}: {e}")
        return None

def process_cifar100_images(split='train'):
    """
    Process CIFAR-100 images and save them as PNG files.
    
    Parameters:
    split (str): Dataset split to process ('train' or 'test').

    Returns:
    None
    """
    path = f'./data/cifar-100-python/{split}'
    data_dict = unpickle(path)
    meta_dict = unpickle('./data/cifar-100-python/meta')
    
    if data_dict is None or meta_dict is None:
        return
    
    matrix = data_dict[b'data']
    fine_labels_list = data_dict[b'fine_labels']
    coarse_labels_list = data_dict[b'coarse_labels']
    
    df = pd.DataFrame(fine_labels_list, columns=['fine_labels'])
    df['coarse_labels'] = coarse_labels_list
    df['image_num'] = df.index + (100000 if split == 'train' else 200000)
    
    img_folder = f'{split}_images'
    if not os.path.isdir(img_folder):
        os.mkdir(img_folder)

    for i in range(matrix.shape[0]):
        image_id = df.loc[i, 'image_num']
        row = matrix[i]
        ch0, ch1, ch2 = row[:1024], row[1024:2048], row[2048:]
        ch0 = np.reshape(ch0, (32, 32))
        ch1 = np.reshape(ch1, (32, 32))
        ch2 = np.reshape(ch2, (32, 32))
        image = np.dstack((ch0, ch1, ch2))
        fname = f'{image_id}.png'
        dst = os.path.join(img_folder, fname)
        im_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(dst, im_bgr)
