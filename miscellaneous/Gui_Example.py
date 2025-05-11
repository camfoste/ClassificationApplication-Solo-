import os
#print(os.getcwd())
#print(os.listdir())  # Lists all files in the current directory
from model_cnn import *


#Inputting the GUI code RAW file in...
import tkinter as tk
import PIL
from PIL import ImageTk, Image
from tkinter import filedialog
'''
docstring
          
'''

#Creation for opening an image
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
    
    # create a filepath, using filedialog to open it b/t an different list of image types
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*jpeg;*.jpeg;*.png")])

    # Create if stmt, if true, it will open the photo in a certain image.
    if file_path:
        # grabs Image, and stores it into photo along with resizing it.
        photo = Image.open(file_path).resize((300,300))
        picture = ImageTk.PhotoImage(photo)

        img_label.config(image= picture)
        img_label.image = picture



# Creation for opening message.... incomplete.
def classify():
    """
    Placeholder function for classifying the selected image.

    Currently, it only updates the `msg_label` with a placeholder text. This will be updated.

    Future Implementation:
        - Load and preprocess the selected image.
        - Pass the image through a trained CNN model for classification.
        - Display the predicted class in `msg_label`.
    """
    #Where we call for CNN?
    msg_label.config(text="incomplete")

#  Main window with it's name. Can change dimensions & name later

# creating a window
root = tk.Tk()
# dimensions in px
root.geometry("600x600")
# Title of Window
root.title("SnapNLearn")


#  Creating Labels

# Image Label
img_label = tk.Label(root)
img_label.pack()

# Message Label
msg_label = tk.Label(root, text = "")
msg_label.pack()

# Image Button
img_button = tk.Button(root, text= "Select Image", command=open_photo)
img_button.pack()

# Message button
msg_button = tk.Button(root, text= "Classify", command=classify)
msg_button.pack()

root.mainloop()




#Ensure "model.pth" is in the same directory as GUI.py. and in GUI.py, load it like this:

num_classes = 100  # Make sure this matches the model
model = Net(num_classes)
model.load_state_dict(torch.load("model.pth"))
model.eval()

