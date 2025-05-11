import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import datetime
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter
from model_cnn import Net


# displays datetime in YYYYMMDD_HHMMSS format for unique names and easy logging
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

def validate_model(net, val_loader, writer, epoch):
    """
    Evaluates a trained model on the validation dataset.

    Computes the accuracy by comparing model predictions to ground truth labels.
    Logs the validation accuracy to TensorBoard for the given epoch.

    Args:
        net (nn.Module): The trained model to evaluate.
        val_loader (DataLoader): DataLoader for the validation dataset.
        writer (SummaryWriter): TensorBoard writer to log accuracy.
        epoch (int): The current epoch number (for logging purposes).

    Outputs:
        - Displays validation accuracy after evaluation.
        - Logs validation accuracy to TensorBoard under 'Validation Accuracy'.
    """
          
    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    print(f'Validation Accuracy: {val_accuracy:.2f}%')
    writer.add_scalar('Validation Accuracy', val_accuracy, epoch)


def train(batch_size=64, epochs=50, learning_rate=5e-2, valid_split=0.2):
    """
    Trains a convolutional neural network (CNN) on the CIFAR-100 dataset.

    Args:
        batch_size (int, optional): Number of samples per batch. Default: 64.
        epochs (int, optional): Number of training iterations over the dataset. Default: 30.
        learning_rate (float, optional): Step size for optimizer updates. Default: 5e-3.
        valid_split (float, optional): Fraction of training data used for validation. Default: 0.2.

    Steps:
        1. Loads the CIFAR-100 dataset with the specified transformations.
        2. Splits the dataset into training and validation subsets.
        3. Initializes the CNN model.
        4. Sets up logging with TensorBoard.
        5. Defines the loss function (CrossEntropyLoss) and optimizer (SGD).
        6. Trains the model over the specified epochs, updating weights with backpropagation.
        7. Logs loss values and evaluates the model on the validation set.
        8. Saves the trained model to a file.

    Outputs:
        - Prints loss values during training.
        - Displays validation accuracy after training.
        - Saves the trained model with a timestamped filename.
        
    Calling docstring
        -help(train)
    """

    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=False, transform = transform)
              
    # gets dataset size, finds split index
    set_size = len(trainset)
    index = list(range(set_size))
    split = int(valid_split * set_size)

    # splits training set into training (80%) and validation (20%) after shuffling
    torch.manual_seed(0)
    train_id, valid_id = index[split:], index[:split]
    train_sample = SubsetRandomSampler(train_id)
    valid_sample = SubsetRandomSampler(valid_id)    
    
    # breaks dataset into mini batches for training
    trainloader = DataLoader(trainset, batch_size=batch_size, sampler=train_sample, num_workers=0)
    
    val_loader = DataLoader(trainset, batch_size=batch_size, sampler=valid_sample, num_workers=0)
    
    net = Net(num_classes=100)

    # helps log and save data during training for tensorboard        
    writer = SummaryWriter(f'runs/exp_{timestamp}')
    
    # define optimizer and loss
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    
    #training epochs loops with loss
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0

        print(f"\nEpoch {epoch+1}")
        print("-------------------------------")

        for batch, (inputs, labels) in enumerate(trainloader):
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch % 100 == 0:
                print(f'Batch {batch}, Loss: {loss.item():.6f}')
                writer.add_scalar('Training Loss', loss.item(), epoch)    

    # validation step with accuracy
    validate_model(net, val_loader, writer, epoch)

    # close writer, save model
    writer.close()
    model_path = f'model_{timestamp}.pth'
    torch.save(net.state_dict(), model_path)
    print(f'Model saved to {model_path}')
    

if __name__ == '__main__':            
    train() 
