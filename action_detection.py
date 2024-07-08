import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Load the BLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# List of image paths
image_paths = [
    "Images/catPlayingWithABall.jpeg",
    "Images/dogWithBaseballBat.jpeg",
    "Images/Football_Kid.jpg"
]

# ANSI escape sequence for bold text
bold = "\033[1m"
reset = "\033[0m"

# Process each image
for image_path in image_paths:
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Generate a caption for the image
    with torch.no_grad():
        outputs = model.generate(**inputs)
        caption = processor.decode(outputs[0], skip_special_tokens=True)

    # Print image path and bold caption
    print(f"Image: {image_path}")
    print(f"Recognized action: {bold}{caption}{reset}")
    print()
