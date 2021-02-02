import streamlit as st
import plotly.express as px
import numpy as np
import torch
from main import Generator


@st.cache
def load_model():
    device = torch.device('cpu')
    netG = Generator(ngpu=0, ngf=64, nc=3, nz=100)
    netG.load_state_dict(torch.load('./Generator.pth', map_location=device), strict=False)
    netG.eval()

    return netG


st.title('Face Generator | DCGAN')

# Sidebar Configuration
st.sidebar.subheader('Config')
z_value = st.sidebar.slider('Z Value', min_value=0, max_value=1000, step=2)

# Load GAN Model
netG = load_model()

# Generate GAN
torch.manual_seed(z_value)
noise = torch.randn(1, 100, 1, 1, device='cpu')
fake = netG(noise).detach().cpu()

fig = px.imshow(np.transpose(fake[0], (1, 2, 0)))
st.plotly_chart(fig)
