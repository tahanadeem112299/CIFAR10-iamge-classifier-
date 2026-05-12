import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    )
])
class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.norm1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.norm2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.norm3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 512, 3, 1, 1)
        self.norm4 = nn.BatchNorm2d(512)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.3)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(512 * 2 * 2, 1024)

        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):

        x = self.pool(self.relu(self.norm1(self.conv1(x))))

        x = self.pool(self.relu(self.norm2(self.conv2(x))))

        x = self.pool(self.relu(self.norm3(self.conv3(x))))

        x = self.pool(self.relu(self.norm4(self.conv4(x))))

        x = x.view(x.size(0), -1)

        x = self.dropout(x)

        x = self.relu(self.fc1(x))

        x = self.fc2(x)

        return x
model = CNN().to(device)

model.load_state_dict(
    torch.load("cifar_model.pth", map_location=device)
)

model.eval()

root = tk.Tk()
root.title("CIFAR-10 Classifier")
image_label = tk.Label(root)
image_label.pack(pady=20)
result_label = tk.Label(
    root,
    text="CIFAR-10 Classifier",
    font=("Arial", 16)
)

result_label.pack(pady=20)
def upload_image():

    file_path = filedialog.askopenfilename(
        filetypes=[
            ("Image Files", "*.jpg *.png *.jpeg")
        ]
    )

    if file_path:

        image = Image.open(file_path).convert("RGB")

        display_image = image.resize((250, 250))

        photo = ImageTk.PhotoImage(display_image)

        image_label.config(image=photo)

        image_label.image = photo

       
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():

            output = model(image_tensor)

            _, prediction = torch.max(output, 1)

        result_label.config(
            text=f"Uploaded image of : {classes[prediction.item()]}"
        )
upload_button = tk.Button(
    root,
    text="Upload Image",
    command=upload_image,
    font=("Arial", 14),
    padx=20,
    pady=10
)

upload_button.pack()
root.mainloop()