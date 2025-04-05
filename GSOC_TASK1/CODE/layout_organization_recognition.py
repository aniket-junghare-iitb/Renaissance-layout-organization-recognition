import os
from pdf2image import convert_from_path
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
import pytesseract
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import torchvision.models
import torch.optim as optim
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import docx
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import torchvision
import torch.optim as optim
import seaborn as sns
from skimage.metrics import structural_similarity as ssim


PDF_DIR = "/home/aniketj/GSOC_TASK1/PDFs/"   # Directory containing PDFs
IMAGE_DIR = "/home/aniketj/GSOC_TASK1//IMAGES/"     # Output directory for images
os.makedirs(IMAGE_DIR, exist_ok=True)


def pdf_to_images(pdf_path, output_folder, dpi=200):
    #Convert PDF pages to images one by one, reducing image size issues.
    images = convert_from_path(pdf_path, dpi=dpi, fmt="jpeg")  
    image_paths = []
    
    for i, img in enumerate(images):
        img = img.convert("RGB")  
        img_path = os.path.join(output_folder, f"{os.path.basename(pdf_path).replace('.pdf', '')}_page_{i+1}.jpg")
        img.save(img_path, "JPEG", quality=85)  
        image_paths.append(img_path)
    
    return image_paths

for pdf in os.listdir(PDF_DIR):
    if pdf.endswith(".pdf"):
        pdf_to_images(os.path.join(PDF_DIR, pdf), IMAGE_DIR, dpi=200)

print("PDF to Image Conversion Done")


PROCESSED_DIR = "/home/aniketj/GSOC_TASK1/PROCESSED_IMAGES/"
os.makedirs(PROCESSED_DIR, exist_ok=True)

def preprocess_image(image_path, output_folder):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    img = cv2.GaussianBlur(img, (5, 5), 0)  # Apply Gaussian blur for noise reduction
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Otsu's thresholding

    processed_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(processed_path, binary)  # Save processed image
    return processed_path

for img_file in tqdm(os.listdir(IMAGE_DIR), desc="Processing Images"):
    if img_file.endswith(".jpg"):
        preprocess_image(os.path.join(IMAGE_DIR, img_file), PROCESSED_DIR)

print("Image Preprocessing Done")


pytesseract.pytesseract.tesseract_cmd = r'/home/aniketj/anaconda3/envs/soc/bin/tesseract' 

TEXT_REGION_DIR = "/home/aniketj/GSOC_TASK1/TEXT_REGIONS/"
os.makedirs(TEXT_REGION_DIR, exist_ok=True)

def extract_text_regions(image_path, output_folder, visualize=False):

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    d = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    
    for i in range(len(d["text"])):
        if int(d["conf"][i]) > 50:  
            (x, y, w, h) = (d["left"][i], d["top"][i], d["width"][i], d["height"][i])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box

    processed_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(processed_path, img)  

    
    if visualize:
        cv2.imshow("Text Detection", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return processed_path


for img_file in tqdm(os.listdir(PROCESSED_DIR), desc="Extracting Text Regions"):
    if img_file.endswith(".jpg"):
        extract_text_regions(os.path.join(PROCESSED_DIR, img_file), TEXT_REGION_DIR)

print("Text Region Extraction Done")


JSON_OUTPUT = "/home/aniketj/GSOC_TASK1/BOUNDING_BOXES.json"  
bounding_boxes = {}

def extract_bounding_boxes(image_path):
    """Extract text bounding box coordinates using OCR."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    d = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

    boxes = []  

    for i in range(len(d["text"])):
        if int(d["conf"][i]) > 50: 
            x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
            boxes.append({"x": x, "y": y, "width": w, "height": h})

    return boxes




for img_file in tqdm(os.listdir(TEXT_REGION_DIR), desc="Extracting Bounding Boxes"):
    if img_file.endswith(".jpg"):
        img_path = os.path.join(TEXT_REGION_DIR, img_file)
        bounding_boxes[img_file] = extract_bounding_boxes(img_path)

# Save bounding boxes to a JSON file
with open(JSON_OUTPUT, "w") as f:
    json.dump(bounding_boxes, f, indent=4)

print(f"Bounding boxes saved to {JSON_OUTPUT}")




MASK_DIR = "/home/aniketj/GSOC_TASK1/MASKS/"
os.makedirs(MASK_DIR, exist_ok=True)

def create_segmentation_mask(image_path, boxes, mask_output_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = np.zeros_like(img)  

    for box in boxes:
        x1, y1, x2, y2 = box["x"], box["y"], box["x"] + box["width"], box["y"] + box["height"]
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)  


    cv2.imwrite(mask_output_path, mask)

for img_name, boxes in tqdm(bounding_boxes.items(), desc="Creating Masks"):
    img_path = os.path.join(PROCESSED_DIR, img_name)
    mask_path = os.path.join(MASK_DIR, img_name)
    create_segmentation_mask(img_path, boxes, mask_path)

print(f"Masks saved in {MASK_DIR}")



transform = T.Compose([
    T.ToPILImage(),
    T.Resize((512, 512)),
    T.Grayscale(num_output_channels=3),  # Convert 1-channel grayscale to 3-channel RGB
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize for 3 channels
])




class LayoutSegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.img_files = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.img_files)
        
    
    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        # Load Image & Mask
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Ensure both are resized to (512, 512)
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)  # Nearest neighbor for segmentation masks

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) / 255.0  # Normalize to 0-1

        return image, mask


# Create Dataset & DataLoader
dataset = LayoutSegmentationDataset(PROCESSED_DIR, MASK_DIR, transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

print(f"Dataset Ready: {len(dataset)} images with segmentation masks")



class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])  

        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)  

    
    def forward(self, x):
        x = self.encoder(x)
        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.upconv3(x)
        x = self.final_conv(x)
    
    
        x = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)
    
        return torch.sigmoid(x) 
    

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)

def dice_loss(pred, target, smooth=1.0):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))

criterion = lambda pred, target: 0.5 * nn.BCELoss()(pred, target) + 0.5 * dice_loss(pred, target)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 500

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, masks in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(dataloader)}")

print("Training Complete!")



torch.save(model.state_dict(), "/home/aniketj/GSOC_TASK1/layout_recognition_model.pth")
print("Model saved successfully!")


TEST_IMAGES_DIR = "/home/aniketj/GSOC_TASK1/TEST_IMAGES/"  # Folder with test images
RESULTS_DIR = "/home/aniketj/GSOC_TASK1/PREDICTION_RESULT/"  # Output folder for predictions
os.makedirs(RESULTS_DIR, exist_ok=True)

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((512, 512)),
    T.Grayscale(num_output_channels=3),  # Convert 1-channel grayscale to 3-channel RGB
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize for 3 channels
])


def predict_layout(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not read {image_path}")
        return

    img_resized = transform(img).unsqueeze(0).to(device)  
    with torch.no_grad():
        pred_mask = model(img_resized)  

    pred_mask = pred_mask.squeeze().cpu().numpy()  
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255  

    result_path = os.path.join(RESULTS_DIR, os.path.basename(image_path))
    cv2.imwrite(result_path, pred_mask)
    print(f"Saved predicted mask: {result_path}")


for img_file in os.listdir(TEST_IMAGES_DIR):
    if img_file.endswith(".jpg") or img_file.endswith(".png"):
        predict_layout(os.path.join(TEST_IMAGES_DIR, img_file))


OVERLAY_DIR = "/home/aniketj/GSOC_TASK1/OVERLAYS/"  
os.makedirs(OVERLAY_DIR, exist_ok=True)


def overlay_prediction(image_path, mask_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        print(f"Could not load {image_path} or {mask_path}")
        return

    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(img, 0.7, mask, 0.3, 0)
    overlay_path = os.path.join(OVERLAY_DIR, os.path.basename(image_path))
    cv2.imwrite(overlay_path, overlay)
    print(f"Overlay saved: {overlay_path}")

# Process all images
for img_file in os.listdir(TEST_IMAGES_DIR):
    if img_file.endswith(".jpg") or img_file.endswith(".png"):
        mask_file = os.path.join(RESULTS_DIR, img_file)
        overlay_prediction(os.path.join(TEST_IMAGES_DIR, img_file), mask_file)


PREDICTED_MASKS_DIR = "/home/aniketj/GSOC_TASK1/PREDICTED_MASKS/"  # Output folder for predictions
TESTING_DIR = "/home/aniketj/GSOC_TASK1/PROCESSED_IMAGES"
os.makedirs(PREDICTED_MASKS_DIR, exist_ok=True)
def predict_layout(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not read {image_path}")
        return

    img_resized = transform(img).unsqueeze(0).to(device)  
    with torch.no_grad():
        pred_mask = model(img_resized)  

    pred_mask = pred_mask.squeeze().cpu().numpy()  
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255  

    result_path = os.path.join(PREDICTED_MASKS_DIR, os.path.basename(image_path))
    cv2.imwrite(result_path, pred_mask)
    print(f"Saved predicted mask: {result_path}")


for img_file in os.listdir(TESTING_DIR):
    if img_file.endswith(".jpg") or img_file.endswith(".png"):
        predict_layout(os.path.join(TESTING_DIR, img_file))






