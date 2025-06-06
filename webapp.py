import db
import math
import torch
import model
import torch.nn.functional as F
import streamlit as st
from PIL import Image
from datetime import datetime
from torchvision import transforms
from streamlit_drawable_canvas import st_canvas

# print(model)

if "prediction" not in st.session_state:
  st.session_state.prediction = "--"
if "confidence" not in st.session_state:
  st.session_state.confidence = "--"
if "label" not in st.session_state:
  st.session_state.label = None
if "logs" not in st.session_state:
  st.session_state.logs = db.fetch_logs()
if "feedback" not in st.session_state:
  st.session_state.feedback = ""

def guess():
  if st.session_state.label == None:
    st.session_state.feedback = "Please enter a label"
    return

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

  numberguess = model.load()

  with torch.no_grad():
    output = numberguess(img_tensor)
    probabilities = F.softmax(output, dim=1)
    st.session_state.prediction = torch.argmax(probabilities, dim=1).item()
    st.session_state.confidence = math.trunc(
      probabilities[0][st.session_state.prediction].item() * 100
    )
    # print(f'Prediction: {st.session_state.prediction}')
    # save prediction to postgresql
    db.log({
      'timestamp': datetime.now(),
      'prediction': st.session_state.prediction,
      'label': st.session_state.label
    })
    st.session_state.logs = db.fetch_logs()

  st.session_state.label = None
  st.session_state.feedback = ""

st.title('Digit Recogniser')
st.write('Draw a digit:')

col1, col2 = st.columns(2)

with col1:
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

with col2:
  st.write('## Prediction:', st.session_state.prediction)

  st.write(f'Confidence: {st.session_state.confidence}%')

  label = st.number_input(
    'True label:',
    value=None,
    min_value=0,
    max_value=9,
    key="label"
  )

  button = st.button("Guess", on_click=guess)

  st.write(st.session_state.feedback)

# db.create_logs_table()

# print(logs)

logs_table = """
---
## History
| Timestamp | Pred | Label |
|-----------|------|-------|
"""
for row in st.session_state.logs:
    _, timestamp, prediction, label = row
    timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    logs_table += f"| {timestamp} | {prediction} | {label} |\n"

# Display in Streamlit
st.markdown(logs_table)
