import math
import torch
import torch.nn.functional as F
import streamlit as st
from PIL import Image
from numberguess import model
from torchvision import transforms
from streamlit_drawable_canvas import st_canvas

# print(model)

if "prediction" not in st.session_state:
  st.session_state.prediction = "--"
if "confidence" not in st.session_state:
  st.session_state.confidence = "--"

def guess():
  print('Guessing...')
  if canvas_state.image_data is not None:
    img = Image.fromarray(canvas_state.image_data.astype("uint8"), mode="RGBA")
    img = img.resize((28, 28))
    img = img.convert("L")
  else:
    st.write('Error while reading user drawing input')

  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
  ])

  img_tensor = transform(img).unsqueeze(0)

  model.eval()

  with torch.no_grad():
    output = model(img_tensor)
    probabilities = F.softmax(output, dim=1)
    st.session_state.prediction = torch.argmax(probabilities, dim=1).item()
    st.session_state.confidence = math.trunc(probabilities[0][st.session_state.prediction].item() * 100)
    print(f'Prediction: {st.session_state.prediction}')



st.title('Digit Recogniser')

st.write('Draw a digit:')
canvas_state = st_canvas(
  stroke_width=10,
  background_color='#000000',
  fill_color='#FFFFFF',
  stroke_color='#FFFFFF',
  width=300,
  height=300,
  drawing_mode='freedraw',
  key='canvas'
)
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
