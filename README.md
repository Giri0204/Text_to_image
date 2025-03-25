# Text_to_image
Text-to-Image Conversion using Stable Diffusion

Overview

This project utilizes Stable Diffusion, a powerful text-to-image generation model, to generate images based on textual prompts. It runs on Google Colab using a GPU for faster processing.

Features

Generates high-quality images from text prompts.

Utilizes Stable Diffusion (v1-4) from Hugging Face.

Runs efficiently on a GPU (NVIDIA).

Prerequisites

Ensure you have a Google account and access to Google Colab.

Installation & Setup

Open Google Colab.

Copy and paste the following code into a new Colab notebook:
from google.colab import drive
drive.mount('/content/drive')

!nvidia-smi
!pip install diffusers

# Make sure you're logged in with `huggingface-cli login`

from torch import autocast
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    use_auth_token=True
).to("cuda")

prompt = "a photo of a car"
with autocast("cuda"):
    image = pipe(prompt)["images"][0]

image.save("car_ride.png")
!nvidia-smi


How to Use

Mount Google Drive: Run the first two lines to mount your Google Drive for saving images.

Check GPU Availability: !nvidia-smi confirms GPU access.

Install Dependencies: !pip install diffusers installs the required library.

Log in to Hugging Face:

Run huggingface-cli login in a Colab cell.

Enter your Hugging Face API token (available from Hugging Face).

Run the script:

Enter a text prompt (e.g., "a photo of a car").

The model will generate an image based on the prompt.

The image is saved as car_ride.png.

Check GPU Usage: Running !nvidia-smi at the end helps monitor GPU memory usage.

Example Output

An image named car_ride.png will be generated in the current working directory.
