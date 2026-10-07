import torch
import cv2
import numpy as np
from src.model import SolarUNet

IMAGE_PATH = "data/MAGFiLO_1.0_Kaggle_2026/train/train_images/20140609195854Bh.jpeg"
OUTPUT_PATH = "prediction.png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SolarUNet(num_classes=5)

model.load_state_dict(
    torch.load(
        "checkpoints/solar_unet_baseline.pth",
        map_location=DEVICE
    )
)

model.to(DEVICE)
model.eval()

image = cv2.imread(
    IMAGE_PATH,
    cv2.IMREAD_GRAYSCALE
)

image = image.astype(np.float32) / 255.0
image = torch.from_numpy(image)
image = image.unsqueeze(0).unsqueeze(0)
image = image.to(DEVICE)

with torch.no_grad():
    output = model(image)
    prediction = torch.argmax(output, dim=1)

prediction = prediction[0].cpu().numpy()

prediction_image = (prediction * 50).astype(np.uint8)

cv2.imwrite(
    OUTPUT_PATH,
    prediction_image
)

print("Prediction saved!")
print("Output:", OUTPUT_PATH)
print("Unique classes:", np.unique(prediction))