import streamlit as st
from fastai.vision.all import *
from fastai.learner import CastToTensor
import plotly.express as px
from pathlib import Path
import platform

plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

st.title('Jonzotlarni klassifikatsiya qiluvchi model')
st.text('Ushbu model 4 turdagi jonzotlarni o\'z ichiga oladi\n 1.Yirtqichlar 2.Hashorotlar  3.Reptilyalar 4. Qushlar')

file = st.file_uploader('Upload', type=['png','jpeg','gif','svg'])

if file:

    st.image(file)

    img = PILImage.create(file)

    model = load_learner('animals_model.pkl')

    pred, pred_id, probs = model.predict(img)
    st.success(f'Bashorat: {pred}')
    st.info(f'Ehtimollik: {probs[pred_id]*100:.1f}%')

    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)
