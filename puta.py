import torch
import torchvision.transforms as transforms
from PIL import Image
import streamlit as st
import custom_densenets
import torchvision.transforms.functional as F

# Load the model
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

model = custom_densenets.se_densenet121_model(5)
model.load_state_dict(torch.load('models/SE_DenseNet121_42.ckpt', map_location=torch.device(device)))


# Set the model to evaluation mode
model.eval()

# Define the preprocessing steps for the input image
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ColorJitter(brightness=0),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the function to make predictions
def predict(image):
    # Preprocess the input image
    input_tensor = transform(image)
    #gray_tensor = F.rgb_to_grayscale(input_tensor)
    # Add a batch dimension to the input tensor
    input_tensor = input_tensor.unsqueeze(0)

    # Make predictions
    with torch.no_grad():
        output = model(input_tensor)

    # Convert output to probabilities using softmax function
    probabilities = torch.nn.functional.softmax(output, dim=1)

    # Get the predicted class
    _, predicted_class = torch.max(probabilities, dim=1)

    # Return the predicted class
    return predicted_class.item()

# Define the Streamlit app
icon = Image.open("pp.jpg")
st.set_page_config(
    page_title="Severity Analysis of Arthrosis in the Knee",
    page_icon=icon
)
with st.sidebar:
    st.image(icon)
    st.subheader("Intelligent Radiology assistant")
    st.subheader(":arrow_up: Upload image")
    uploaded_file = st.file_uploader("Choose x-ray image")
    st.sidebar.write("")



# Add a file uploader widget
#uploaded_file = st.file_uploader("Choose an image...", type="png")

# Make predictions when the user uploads an image
if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)

    # Display the image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Make predictions
    prediction = predict(image)

    # Display the predicted class
    st.write(f"Prediction is {prediction}",style={"font-size": "40px"})

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to bottom right, #0000FF, #ffffff);
    }
    .stButton>button {
        background-color: #2f3e98;
        color: white;
        border-color: #2f3e98;
        border-radius: 20px;
        font-size: 20px;
        padding: 10px 20px;
        margin-top: 10px;
    }
    .stTextInput>div>div>input {
        border-radius: 20px;
        padding: 10px 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)