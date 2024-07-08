import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Load the BLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Load and preprocess the image
image_path = "Images/catPlayingWithABall.jpeg"  
image = Image.open(image_path)
inputs = processor(images=image, return_tensors="pt").to(device)

# Generate a caption for the image
with torch.no_grad():
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)

print(f"Recognized action: {caption}")
