import sys 
sys.path.append("/home/akriti/Notebooks/ImageBind/") 
import random
import gradio as gr
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModelForZeroShotImageClassification
from qdrant_client import QdrantClient
from qdrant_client.http import models
import tempfile
import os
from tqdm import tqdm

# Initialize Qdrant client and load collection
client = QdrantClient(":memory:")
device = "cuda"
import imagebind
from imagebind.models import imagebind_model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

import os

# Directory containing the images
directory = './fashion-images/data/Apparel_Boys/Images/images_with_product_ids/'

# Initialize an empty list to store the image paths
image_paths_Boys = []

# Iterate through all files and directories within the given directory
for root, dirs, files in os.walk(directory):
    for file in files:
        # Check if the file has an image extension (e.g., .jpg, .png, .jpeg, etc.)
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            # Construct the full path to the image file
            image_path = os.path.join(root, file)
            # Append the image path to the list
            image_paths_Boys.append(image_path)

from imagebind.models.imagebind_model import ModalityType
from imagebind import data
inputs1 = {ModalityType.VISION: data.load_and_transform_vision_data(image_paths_Boys, device)}
import torch
#generating embeddings
with torch.no_grad():
    embeddings1 = model(inputs1)

import os

# Directory containing the images
directory = './fashion-images/data/Apparel_Girls/Images/images_with_product_ids/'

# Initialize an empty list to store the image paths
image_paths_Girls = []

# Iterate through all files and directories within the given directory
for root, dirs, files in os.walk(directory):
    for file in files:
        # Check if the file has an image extension (e.g., .jpg, .png, .jpeg, etc.)
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            # Construct the full path to the image file
            image_path = os.path.join(root, file)
            # Append the image path to the list
            image_paths_Girls.append(image_path)

from imagebind.models.imagebind_model import ModalityType
from imagebind import data
inputs2 = {ModalityType.VISION: data.load_and_transform_vision_data(image_paths_Girls, device)}
import torch
#generating embeddings
with torch.no_grad():
    embeddings2 = model(inputs2)

import os

# Directory containing the images
directory = './fashion-images/data/Footwear_Men/Images/images_with_product_ids/'

# Initialize an empty list to store the image paths
image_paths_Men = []

# Iterate through all files and directories within the given directory
for root, dirs, files in os.walk(directory):
    for file in files:
        # Check if the file has an image extension (e.g., .jpg, .png, .jpeg, etc.)
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            # Construct the full path to the image file
            image_path = os.path.join(root, file)
            # Append the image path to the list
            image_paths_Men.append(image_path)

from imagebind.models.imagebind_model import ModalityType
from imagebind import data
inputs3 = {ModalityType.VISION: data.load_and_transform_vision_data(image_paths_Men, device)}
import torch
#generating embeddings
with torch.no_grad():
    embeddings3 = model(inputs3)

import os

# Directory containing the images
directory = './fashion-images/data/Footwear_Women/Images/images_with_product_ids/'

# Initialize an empty list to store the image paths
image_paths_Women = []

# Iterate through all files and directories within the given directory
for root, dirs, files in os.walk(directory):
    for file in files:
        # Check if the file has an image extension (e.g., .jpg, .png, .jpeg, etc.)
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            # Construct the full path to the image file
            image_path = os.path.join(root, file)
            # Append the image path to the list
            image_paths_Women.append(image_path)

from imagebind.models.imagebind_model import ModalityType
from imagebind import data
inputs4 = {ModalityType.VISION: data.load_and_transform_vision_data(image_paths_Women, device)}
import torch
#generating embeddings
with torch.no_grad():
    embeddings4 = model(inputs4)

client.recreate_collection(collection_name = "imagebind_data", 
                           vectors_config = {"image": models.VectorParams( size = 1024, distance = models.Distance.COSINE ) } )

def process_text(image_query):
    
    user_query = [image_query]
    dtype, modality = ModalityType.VISION, 'image'
    user_input = {dtype: data.load_and_transform_vision_data(user_query, device)}

    with torch.no_grad():
        user_embeddings = model(user_input)
     
    image_hits = client.search(
        collection_name='imagebind_data',
        query_vector=models.NamedVector(
            name="image",
            vector=user_embeddings[dtype][0].tolist()
            )
    )
    # Check if 'path' is in the payload of the first hit
    if image_hits and 'path' in image_hits[0].payload:
        return (image_hits[0].payload['path'])
    else:
        return None

all_image_paths = []
all_image_paths.append(image_paths_Boys)
all_image_paths.append(image_paths_Girls)
all_image_paths.append(image_paths_Men)
all_image_paths.append(image_paths_Women)

embeddings_list = [embeddings1,embeddings2,embeddings3,embeddings4]

import uuid

points = []

# Iterate over each embeddings and corresponding image paths
for idx, (embedding, image_paths) in enumerate(zip(embeddings_list, all_image_paths)):
    for sub_idx, sample in enumerate(image_paths):
        # Convert the sample to a dictionary
        payload = {"path": sample}
        # Generate a unique UUID for each point
        point_id = str(uuid.uuid4())
        points.append(models.PointStruct(id=point_id,
                                         vector= {"image": embedding['vision'][sub_idx]}, 
                                         payload=payload)
                      )

client.upsert(collection_name="imagebind_data", points=points)

import tempfile
tempfile.tempdir = "./fashion-images/data"

# Gradio Interface
iface = gr.Interface(
    title="Reverse Image Search with Imagebind",
    description="Leveraging Imagebind to perform reverse image search for ecommerce products",
    fn=process_text,
    inputs=[
        gr.Image(label="image_query", type="filepath")
        ],
    outputs=[
        gr.Image(label="Image")],  
)

def get_images_from_category(category):
    # Convert category to string
    category_str = str(category)
    # Directory path for selected category
    category_dir = f"./fashion-images/data/{category_str.replace(' ', '_')}/Images/images_with_product_ids/"
    # List of image paths
    image_paths = os.listdir(category_dir)[:5]
    # Open and return images
    images = [Image.open(os.path.join(category_dir, img_path)) for img_path in image_paths]
    return images


# Define your product categories
product_categories = ["Apparel Boys", "Apparel Girls", "Footwear Men", "Footwear Women"]

# Define function to handle category selection
def select_category(category):
    # Get images corresponding to the selected category
    images = get_images_from_category(category)
    # Return a random image from the list
    return random.choice(images)

category_dropdown = gr.Dropdown(product_categories, label="Select a product category")
images_output = gr.Image(label="Images of Selected Category")

category_search_interface = gr.Interface(
    fn=select_category,
    inputs=category_dropdown,
    outputs=images_output,
    title="Category-driven Product Search for Ecommerce",
    description="Select a product category to view a random image from the corresponding directory.",
)

# Combine both interfaces into the same API
combined_interface = gr.TabbedInterface([iface, category_search_interface])

# Launch the combined interface
combined_interface.launch(share=True)
