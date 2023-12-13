import streamlit as st
from PIL import Image
import torch
from gan_module import Generator
from PIL import Image
from torchvision import transforms
import numpy as np


#Title
st.title('Face Aging Using GANs')

model = Generator(ngf=32, n_residual_blocks=9)
model.load_state_dict(torch.load('pretrained_model/state_dict.pth'))
trans = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

#drag and drop

file = st.file_uploader("Upload file", type=["jpg","jpeg", "png"])

#display image


def produce_image(image):
    img = image.convert('RGB')
    img = trans(img).unsqueeze(0)
    aged_face = model(img)
    aged_face = (aged_face.squeeze().permute(1, 2, 0).detach().numpy() + 1.0) / 2.0

    # # # Plot the original and aged faces
    # # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # # ax[0].imshow((img.squeeze().permute(1, 2, 0).detach().numpy() + 1.0) / 2.0)
    # # ax[0].set_title('Original Face')
    # # ax[1].imshow(aged_face)
    # # ax[1].set_title('Aged Face')

    return aged_face 

if file is not None:
    image = Image.open(file)

    col1, col2 = st.columns(2)

    # st.image(image, caption='Uploaded Image.')
    st.write("")
    st.write("Processing.........")

    old_age_image = produce_image(image)
    if isinstance(old_age_image, np.ndarray):
        old_age_image = (old_age_image * 255).astype(np.uint8)
        old_age_image = Image.fromarray(old_age_image)
    old_age_image = old_age_image.resize(image.size)

    col1.image(image, caption='Uploaded Image.')
    col2.image(old_age_image, caption='Aged Image.')

    # st.image(old_age_image, caption='Aged Image.')
    



    

