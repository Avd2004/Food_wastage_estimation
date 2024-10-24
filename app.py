import streamlit as st
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from transformers import AutoModel
from model import DeeperAutoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the model (assuming it's a PyTorch model)
@st.cache_resource
def load_model():
    model = DeeperAutoencoder()
    model_url = "https://huggingface.co/Avadh4/food_wastage/resolve/main/food_wastage_estimation.pth"
    
    # Download and load model state_dict
    state_dict = torch.hub.load_state_dict_from_url(model_url, map_location=torch.device('cpu'))
    
    # Load the state_dict into the model
    model.load_state_dict(state_dict)
    
    model.eval()  # Set model to evaluation mode
    return model
    # model = AutoModel.from_pretrained('Avadh4/food_wastage')
    # model.load_state_dict()

    # # model_path = "Avadh4/food_wastage/food_wastage_estimation.pth"
    # # model = torch.load(model_path, map_location=torch.device('cpu'),weights_only=True)
    # # model.eval()
    # # return model

# Function to calculate the difference and estimate food wastage
def calculate_difference(plate_with_food, model, threshold=0.1):
    with torch.no_grad():
        plate_with_food = plate_with_food.unsqueeze(0).to(device)  # Add batch dimension
        reconstructed_empty_plate = model(plate_with_food)
        
        # Calculate the difference
        difference = torch.abs(plate_with_food - reconstructed_empty_plate)
        binary_difference = (difference > threshold).float()  # Identify significant changes
        
        wastage_pixels = binary_difference.sum().item()  # Number of pixels above the threshold
        total_pixels = binary_difference.numel()  # Total number of pixels
        
        # Calculate the percentage of the plate area covered by food wastage
        wastage_percentage = (wastage_pixels / total_pixels) * 100
        
        return wastage_pixels, wastage_percentage, reconstructed_empty_plate, difference

# Function to visualize original, reconstructed, and difference
def visualize_difference(image, model, transform):
    image_tensor = transform(image).to(device)
    wastage_pixels, wastage_percentage, reconstructed, difference = calculate_difference(image_tensor, model)

    # Show output in streamlit
    st.write(f"Estimated wastage area (number of pixels): {wastage_pixels}")
    st.write(f"Estimated wastage area as percentage of the plate: {wastage_percentage:.2f}%")
    
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(image_tensor.cpu().permute(1, 2, 0).numpy())
    axs[0].set_title('Plate with wastage')
    
    axs[1].imshow(reconstructed.cpu().squeeze().permute(1, 2, 0).numpy())
    axs[1].set_title('Reconstructed (Empty Plate)')
    
    axs[2].imshow(difference.cpu().squeeze().permute(1, 2, 0).numpy(), cmap='hot')
    axs[2].set_title('Difference Map (Food Wastage)')
    
    st.pyplot(fig)

# Streamlit app interface
st.title("Food Wastage Estimation")

# Image upload
uploaded_file = st.file_uploader("Choose an image of a plate with food", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    # Image transformation (assuming same transformations as in your model training)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize the image if needed
        transforms.ToTensor()  # Convert the image to a tensor
    ])
    
    st.image(image, caption="Uploaded Plate Image", use_column_width=True)
    
    # Load model
    model = load_model()
    
    # Perform estimation and visualization
    visualize_difference(image, model, transform)
else:
    st.write("Please upload an image.")
