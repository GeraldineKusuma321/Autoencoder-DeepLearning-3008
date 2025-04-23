import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt

# -------------------------------------
# Input Gambar dari Pengguna
# -------------------------------------
image_path = input("Masukkan path gambar yang ingin diproses: ")

try:
    image = Image.open(image_path).convert("RGB")
except FileNotFoundError:
    print(f"File '{image_path}' tidak ditemukan. Pastikan path yang dimasukkan benar.")
    exit()

# -------------------------------------
# Meminta Tingkat Kecerahan dari Pengguna
# -------------------------------------
while True:
    try:
        brightness_factor = float(input("Masukkan tingkat kecerahan (misalnya 1.8, 2.0, dst.): "))
        if brightness_factor < 0.5 or brightness_factor > 3.0:
            print("Mohon masukkan nilai antara 0.5 hingga 3.0 untuk hasil yang optimal.")
        else:
            break
    except ValueError:
        print("Masukkan angka yang valid!")

# -------------------------------------
# Membuat Dataset Target (Gambar Lebih Terang)
# -------------------------------------
target_dir = "data/target"
os.makedirs(target_dir, exist_ok=True)

output_path = os.path.join(target_dir, os.path.basename(image_path))

# Proses peningkatan kecerahan
enhancer = ImageEnhance.Brightness(image)
bright_img = enhancer.enhance(brightness_factor)
bright_img.save(output_path)

# -------------------------------------
# Model dan Dataset
# -------------------------------------
class SingleImageDataset(torch.utils.data.Dataset):
    def __init__(self, input_img, target_img, transform=None):
        self.input_img = input_img
        self.target_img = target_img
        self.transform = transform

    def __len__(self):
        return 1 

    def __getitem__(self, idx):
        if self.transform:
            input_img = self.transform(self.input_img)
            target_img = self.transform(self.target_img)
        return input_img, target_img

class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 3, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# -------------------------------------
# Training Model dengan Gambar yang Diinputkan
# -------------------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

dataset = SingleImageDataset(image, bright_img, transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleUNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(1, 100):
    model.train()
    running_loss = 0.0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f"Epoch {epoch}, Loss: {running_loss:.4f}")

torch.save(model.state_dict(), "single_image_model.pth")
print("Model telah disimpan sebagai 'single_image_model.pth'.")

# -------------------------------------
# Prediksi dan Visualisasi
# -------------------------------------
try:
    model.load_state_dict(torch.load("single_image_model.pth", map_location=device))
    model.eval()
    print("Model berhasil dimuat.")
except FileNotFoundError:
    print("File model tidak ditemukan.")
    exit()

# Prediksi model
input_tensor = transform(image).unsqueeze(0).to(device)
output_tensor = model(input_tensor).detach().cpu().squeeze(0)

# Konversi tensor ke gambar numpy
input_img = input_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
output_img = output_tensor.permute(1, 2, 0).cpu().numpy()
target_img = transform(bright_img).squeeze(0).permute(1, 2, 0).cpu().numpy()

# Visualisasi hasil
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(input_img)
plt.title("Input")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(output_img)
plt.title("Predicted")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(target_img)
plt.title(f"Target ({brightness_factor})")
plt.axis('off')

plt.show()