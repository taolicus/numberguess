import math
import torch
import torch.nn.functional as F
import streamlit as st
from numberguess import model
from torchvision import transforms
from PIL import Image

# print(model)

if "prediction" not in st.session_state:
  st.session_state.prediction = "--"
if "confidence" not in st.session_state:
  st.session_state.confidence = "--"

def guess():
  print('Guessing...')

  img = Image.open('./img/58654.png').convert('L')

  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
  ])

  img_tensor = transform(img)
  img_tensor = img_tensor.unsqueeze(0)

  model.eval()

  with torch.no_grad():
      output = model(img_tensor)
      probabilities = F.softmax(output, dim=1)
      st.session_state.prediction = torch.argmax(probabilities, dim=1).item()
      st.session_state.confidence = math.trunc(probabilities[0][st.session_state.prediction].item() * 100)

# print(f'Predicted class: {predicted_class}')


st.title('Digit Recogniser')

st.write('Draw a digit:')
st.write('## Prediction:', st.session_state.prediction)
st.write(f'Confidence: {st.session_state.confidence}%')

label = st.number_input('True label:', value=0, min_value=0, max_value=9)

button = st.button("Guess", on_click=guess)

# st.write(
# f"""
# ---

# ## History
# | timestamp | pred | label |
# |-----------|------|-------|
# ||||
# """
# )
