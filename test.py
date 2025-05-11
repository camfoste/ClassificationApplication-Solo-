#import statements
import torch
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as  plt
import torchvision.transforms as transforms
import sklearn
import glob
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from model_cnn import Net
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

fine_to_superclass = {
     0:  4,  1:  1,  2: 14,  3:  8,  4:  0,  5:  6,  6:  7,  7:  7,  8: 18,  9:  3,
    10:  3, 11: 14, 12:  9, 13: 18, 14:  7, 15: 11, 16:  3, 17:  9, 18:  7, 19: 11,
    20:  6, 21: 11, 22:  5, 23: 10, 24:  7, 25:  6, 26: 13, 27: 15, 28:  3, 29: 15,
    30:  0, 31: 11, 32:  1, 33: 10, 34: 12, 35: 14, 36: 16, 37:  9, 38: 11, 39:  5,
    40:  5, 41: 19, 42:  8, 43:  8, 44: 15, 45: 13, 46: 14, 47: 17, 48: 18, 49: 10,
    50: 16, 51:  4, 52: 17, 53:  4, 54:  2, 55:  0, 56: 17, 57:  4, 58: 18, 59: 17,
    60: 10, 61:  3, 62:  2, 63: 12, 64: 12, 65: 16, 66: 12, 67:  1, 68:  9, 69: 19,
    70:  2, 71: 10, 72:  0, 73:  1, 74: 16, 75: 12, 76:  9, 77: 13, 78: 15, 79: 13,
    80: 16, 81: 19, 82:  2, 83:  4, 84:  6, 85: 19, 86:  5, 87:  5, 88:  8, 89: 19,
    90: 18, 91:  1, 92:  2, 93: 15, 94:  6, 95:  0, 96: 17, 97:  8, 98: 14, 99: 13
}

# Data loading
transform = transforms.Compose([transforms.ToTensor()])
testset = datasets.CIFAR100(root='./data', train=False, download=False, transform=transform)

# Gets the most recent model file using glob
model_files = glob.glob('./model_*.pth')
if not model_files:
    raise FileNotFoundError("No model files found.")

# Sorts the files by timestamp (the digits in the file name)
model_files.sort(key=lambda x: x[-15:], reverse=True)

# Select the most recent one
PATH = model_files[0]
print(f"Loading model from {PATH}")


# Helper Functions:

def create_file(filename,pred,recall,f1):
    '''
    Writes precision, recall, and F1-score values for each class to a text file.

    Parameters
    ----------
    filename : str
        Name of the file to create and write the metrics into.

    pred : list of float
        Precision values corresponding to each class.

    recall : list of float
        Recall values corresponding to each class.

    f1 : list of float
        F1-score values corresponding to each class.

    Returns
    -------
    None
        Writes data to the specified file and prints a success message.
    '''
    with open(filename, 'w') as file:
        file.write("Index | Precision | Recall | F1\n")
        for i, (p,r,f) in enumerate(zip(pred,recall,f1)):
            file.write(f'{i} ,{p:.2f} ,{r:.2f} ,{f:.2f}\n')
    print(f'File "{filename}" has been created')

def acc_per_class(n, labels, preds):
    '''
    Calculates the accuracy for each class (or superclass) individually.

    Parameters
    ----------
    n : int
        Number of classes. Use 100 for fine-grained classes or 20 for superclasses.

    labels : torch.Tensor
        Ground truth labels for the dataset.

    preds : torch.Tensor
        Model predictions corresponding to each input.

    Returns
    -------
    list of float
        A list containing per-class accuracy values.
    '''
    
    acc = [0.0] * n

    if n == 100:
        for i in range(100):
            correct = 0
            total = 0
            for l, p in zip(labels, preds):
                if l == i:
                    total += 1
                    if p == i:
                        correct += 1
            if total > 0:
                acc[i] = correct / total
            else:
                acc[i] = 0.0

    elif n == 20:
        for i in range(20):
            correct = 0
            total = 0
            for l, p in zip(labels, preds):
                if fine_to_superclass[l.item()] == i:
                    total += 1
                    if fine_to_superclass[p.item()]== fine_to_superclass[l.item()]:
                        correct += 1
            if total > 0:
                acc[i] = correct / total
            else:
                acc[i] = 0.0
    return acc


def metrics(all_preds, all_labels):
    '''
    Computes the accuracy, percision,recall, and F-1 score metrics

    Parameters
    ----------
        all_preds(Tensor): Predicted Labels
        all_labels)Tensor): True labels

    Returns
    -------
        Accuracy
        per-class precision/recall/F1-score
        macro-metrics

    '''
    acc_score = accuracy_score(all_labels.cpu(), all_preds.cpu())
    pred = precision_score(all_labels.cpu(), all_preds.cpu(), average=None, zero_division=0)
    recall = recall_score(all_labels.cpu(), all_preds.cpu(), average=None, zero_division=0)
    f1 = f1_score(all_labels.cpu(), all_preds.cpu(), average=None, zero_division=0)

    mac_pred = precision_score(all_labels.cpu(), all_preds.cpu(), average='macro')
    mac_recall = recall_score(all_labels.cpu(), all_preds.cpu(), average='macro')
    mac_f1 = f1_score(all_labels.cpu(), all_preds.cpu(), average='macro')
    return acc_score, pred, recall, f1, mac_pred, mac_recall, mac_f1

def calculate_superclass_labels(labels_tensor):
    '''
    Maps CIFAR-100 fine classes to their superclass

    Parameters
    --------
        labels_tensor(tensor): fine class labels
    Returns:
    --------
        Tensor: corresponding superclass labels

    '''
    return torch.tensor([fine_to_superclass[i.item()] for i in labels_tensor])

def print_metrics(all_preds, all_labels):
    '''
    Prints the metrics for both fine and super class and gets the information from the metrics function

    Parameters
    -----
        all_preds (Tensor): Predicted fine class labels
        all_labels (Tensor): Ground truth fine class labels
    Returns:
    ----
        Doesn't return, but prints out the metrics.      
    '''
    acc_score, pred, recall, f1, mac_pred, mac_recall, mac_f1 = metrics(all_preds, all_labels)
    print("\n---Metrics---")
    print(f'\nMean Accuracy: {acc_score:.2f}')

    print("\nAccuracy Per Class:")
    print('\n'.join(
        f'Class {i:2d} : {a:.2f}' for i, a in enumerate(acc_per_class(100, all_labels, all_preds))))

    print(f'\nMacro Precision: {mac_pred:.2f}')
    print(f'Macro Recall: {mac_recall:.2f}')
    print(f'Macro F1-Score: {mac_f1:.2f}')
    
    print("\nIndex | Precision | Recall  | F1")
    print("-" * 35)
    print("\n".join(
        f"{i:2d} | {p: .2f}| {r:.2f} | {f:.2f}"
        for i, (p, r, f) in enumerate(zip(pred, recall, f1))))

    create_file("Class List.txt", pred, recall, f1)

    # Superclass metrics
    super_preds = calculate_superclass_labels(all_preds)
    super_labels = calculate_superclass_labels(all_labels)

    super_acc = accuracy_score(super_labels, super_preds)
    super_pred = precision_score(super_labels, super_preds, average=None, zero_division=0)
    super_recall = recall_score(super_labels, super_preds, average=None, zero_division=0)
    super_f1 = f1_score(super_labels, super_preds, average=None, zero_division=0)

    print("\n---Superclass Metrics---")
    print(f'\nSuperclass Accuracy: {super_acc:.2f}')
    print("\nAccuracy Per SuperClass:")
    print('\n'.join(
        f'Class {i:2d} : {a:.2f}' for i, a in enumerate(acc_per_class(20, super_labels, super_preds))))

    print(f'Macro Superclass Precision: {precision_score(super_labels, super_preds, average="macro", zero_division=0):.2f}')
    print(f'Macro Superclass Recall: {recall_score(super_labels, super_preds, average="macro", zero_division=0):.2f}')
    print(f'Macro Superclass F1-Score: {f1_score(super_labels, super_preds, average="macro", zero_division=0):.2f}')

    print("\nIndex | Precision | Recall  | F1")
    print("-" * 35)
    print("\n".join(
        f"{i:2d} | {p: .2f}| {r: .2f} | {f: .2f}"
        for i, (p, r, f) in enumerate(zip(super_pred, super_recall, super_f1))))

    create_file("SuperClass List.txt", super_pred, super_recall, super_f1)



# Main function                                 
def test(dataloader=
         DataLoader(testset,batch_size=64, shuffle=False),
         model=Net(num_classes=100),
         loss_fn=nn.CrossEntropyLoss()): #put parameters as required
    """
    Evaluates the performance of a trained CNN model on the CIFAR-100 test dataset.

    This function loads the most recent model checkpoint, uses the provided DataLoader 
    to pass test images through the model, and calculates several evaluation metrics.
    It reports accuracy, precision, recall, and F1-score for both fine-level and superclass 
    classification, including macro-averaged metrics This happens with the help of the helper functions

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader, optional
        DataLoader for the test dataset. Defaults to CIFAR-100 test set with batch_size=64.
    
    model : torch.nn.Module, optional
        CNN model to be evaluated. Defaults to an instance of `Net` with 100 output classes.

    loss_fn : torch.nn.modules.loss._Loss, optional
        Loss function used for evaluation (not actively used for metrics here, but retained for consistency).
        Defaults to `nn.CrossEntropyLoss()`.

    Returns
    -------
    None
        This function does not return any values. Instead, it prints:
        - Fine-grained classification metrics: accuracy, precision, recall, F1-score
        - Macro-averaged versions of precision, recall, and F1-score
        - Superclass-level metrics: accuracy, precision, recall, F1-score

    Notes
    -----
    - Uses a mapping from CIFAR-100 fine labels to superclass labels for additional evaluation.
    - Assumes the presence of a trained model file in the current directory with a name pattern `model_*.pth`.
    - The most recent model (based on filename timestamp) is used for evaluation.
    """
    
    # load the trained model using the path
    model.load_state_dict(torch.load(PATH))
    model.eval()

    size = len(dataloader.dataset)   
    num_batches = len(dataloader)

    correct = 0 # Track of amt of corrected Images
    super_correct = 0
    total = 0 # Track of amt of Images processed
    # other initializations for metric computation
    all_preds=[]
    all_labels=[]

    with torch.no_grad():# No gradient so that we don't update our weights and biases
        for i, data in enumerate(dataloader):
            images,labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data,1)
            total += labels.size()[0]
            correct += (predicted == labels).sum().item()


            all_preds.append(predicted)
            all_labels.append(labels)

    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    print_metrics(all_preds,all_labels)
    
if __name__ == '__main__':         
    test() 
