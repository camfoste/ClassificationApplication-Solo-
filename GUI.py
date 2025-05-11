import os
import torch
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
from model_cnn import Net
import glob

#Addition:
from torchvision import transforms

# Define class names and superclasses
class_names = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'cra', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel',
    'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

superclass_names = { 
    0: "aquatic_mammals", 1: "fish", 2: "flowers", 3: "food_containers",
    4: "fruit_and_vegetables", 5: "household_electrical_devices", 6: "household_furniture",
    7: "insects", 8: "large_carnivores", 9: "large_man-made_outdoor_things",
    10: "large_natural_outdoor_scenes", 11: "large_omnivores_and_herbivores", 12: "medium_mammals",
    13: "non-insect_invertebrates", 14: "people", 15: "reptiles", 16: "small_mammals",
    17: "trees", 18: "vehicles_1", 19: "vehicles_2"
}

fine_to_super = {
    0: 4, 1: 1, 2: 14, 3: 8, 4: 0, 5: 6, 6: 7, 7: 7, 8: 18, 9: 3,
    10: 3, 11: 14, 12: 9, 13: 18, 14: 7, 15: 11, 16: 3, 17: 9, 18: 7, 19: 11,
    20: 6, 21: 11, 22: 5, 23: 10, 24: 7, 25: 6, 26: 13, 27: 15, 28: 3, 29: 15,
    30: 0, 31: 11, 32: 1, 33: 10, 34: 12, 35: 14, 36: 16, 37: 9, 38: 11, 39: 5,
    40: 5, 41: 19, 42: 8, 43: 8, 44: 15, 45: 13, 46: 14, 47: 17, 48: 18, 49: 10,
    50: 16, 51: 4, 52: 17, 53: 4, 54: 2, 55: 0, 56: 17, 57: 4, 58: 18, 59: 17,
    60: 10, 61: 3, 62: 2, 63: 12, 64: 12, 65: 16, 66: 12, 67: 1, 68: 9, 69: 19,
    70: 2, 71: 10, 72: 0, 73: 1, 74: 16, 75: 12, 76: 9, 77: 13, 78: 15, 79: 13,
    80: 16, 81: 19, 82: 2, 83: 4, 84: 6, 85: 19, 86: 5, 87: 5, 88: 8, 89: 19,
    90: 18, 91: 1, 92: 2, 93: 15, 94: 6, 95: 0, 96: 17, 97: 8, 98: 14, 99: 13
}

# Load trained model
model_files = glob.glob('./model_*.pth')
if not model_files:
    raise FileNotFoundError("No model files found.")
    
model_files.sort(key=lambda x: x[-15:], reverse=True)
model_path = model_files[0]

num_classes = 100
model = Net(num_classes)
model.load_state_dict(torch.load(model_path))
model.eval()

# Function to run model on selected image
def run_model(image_path):
    """
    Runs a trained CNN model on the provided image and returns the predicted label.

    Args:
        image_path (str): Path to the image file to be classified.

    Returns:
        str: Predicted label from the CIFAR-100 dataset.
    """
  #Addition:
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

    # Open the image, convert to RGB, and apply transformations
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to("cpu")  # Move tensor to CPU ; End of Addition.

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)

        fine_class = predicted.item()
        fine_class_name = class_names[fine_class]
        super_class_idx = fine_to_super[fine_class]
        super_class_name = superclass_names[super_class_idx]

        return fine_class_name, super_class_name

# GUI functions
def open_photo():
    """
    Opens an image file using a file dialog and displays it in the GUI.

    Steps:
        1. Opens a file dialog for selecting an image file (JPEG or PNG).
        2. If a file is selected, loads the image and resizes it to 300x300 pixels.
        3. Updates the GUI to display the selected image.

    Effects:
        - Updates `img_label` to show the selected image.
    """
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpeg;*.jpg;*.png")])
    if file_path:
        photo = Image.open(file_path).resize((300, 300))
        picture = ImageTk.PhotoImage(photo)
        img_label.config(image=picture)
        img_label.image = picture
        img_label.file_path = file_path

def classify():
    """
    Classifies the currently selected image using the trained model and updates the GUI with the prediction.

    Args:
        None

    Returns:
        None
    """
    if hasattr(img_label, 'file_path'):
        fine_class, super_class = run_model(img_label.file_path)
        msg_label.config(text=f"Prediction:\n{super_class}\n{fine_class}")
    else:
        msg_label.config(text="No image selected.")

# Create GUI
root = tk.Tk()
root.geometry("600x600")
root.title("SnapNLearn")

img_label = tk.Label(root)
img_label.pack()

msg_label = tk.Label(root, text="")
msg_label.pack()

img_button = tk.Button(root, text="Select Image", command=open_photo)
img_button.pack()

msg_button = tk.Button(root, text="Classify", command=classify)
msg_button.pack()

icon_path = os.path.join("SnapNLearn.png")
icon = ImageTk.PhotoImage(Image.open(icon_path))
root.iconphoto(False,icon)


root.mainloop()
