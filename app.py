import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import math
import pandas as pd
import altair as alt
from streamlit_lottie import st_lottie
import requests
import plotly.graph_objects as go
import time
try:
 from PIL import Image
except ImportError:
 import Image

# Load the trained model
model = tf.keras.models.load_model('MusicGenre_CNN.h5')

# Define genre dictionary
genre_dict = {0: "disco", 1: "pop", 2: "classical", 3: "metal", 4: "rock", 5: "blues", 6: "hiphop", 7: "reggae", 8: "country", 9: "jazz"}

# Function to preprocess audio file
def process_input(audio_file, track_duration):
    SAMPLE_RATE = 22050
    NUM_MFCC = 13
    N_FTT = 2048
    HOP_LENGTH = 512
    TRACK_DURATION = track_duration  # measured in seconds
    SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
    NUM_SEGMENTS = 10

    samples_per_segment = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / HOP_LENGTH)

    signal, sample_rate = librosa.load(audio_file, sr=SAMPLE_RATE)

    for d in range(10):
        start = samples_per_segment * d
        finish = start + samples_per_segment
        mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sample_rate, n_mfcc=NUM_MFCC, n_fft=N_FTT, hop_length=HOP_LENGTH)
        mfcc = mfcc.T

    return mfcc, signal, sample_rate

# Streamlit app
st.title("MoodMuse 🎶")
image = Image.open('rock.jpg')
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
anime1="https://assets10.lottiefiles.com/private_files/lf30_fjln45y5.json"
anime1_json=load_lottieurl(anime1)
st_lottie(anime1_json,key='music')

def animated_waveform(audio_file):
    audio_data, sr = librosa.load(audio_file, sr=None)
    duration = librosa.get_duration(y=audio_data, sr=sr)
    frames = np.array_split(audio_data, int(duration * 10))  # split into frames for smoother animation
    
    fig = go.Figure(layout=go.Layout(
        title="Animated Waveform",
        xaxis=dict(title="Time"),
        yaxis=dict(title="Amplitude"),
    ))

    # Initialize an empty line that will be updated
    waveform = go.Scatter(x=[], y=[], mode="lines")
    fig.add_trace(waveform)

    # Render the figure in Streamlit
    plotly_fig = st.plotly_chart(fig, use_container_width=True)
    
    # Animate each frame to update the waveform
    for i, frame in enumerate(frames):
        x_vals = np.linspace(i / 10, (i + 1) / 10, len(frame))
        waveform = go.Scatter(x=x_vals, y=frame, mode="lines", line=dict(color='royalblue'))
        fig.update_traces(overwrite=True, x=x_vals, y=frame)
        plotly_fig.plotly_chart(fig, use_container_width=True)
        time.sleep(0.1)  # Control animation speed

# File uploader
uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

if uploaded_file is not None:
    # Display audio file
    st.audio(uploaded_file, format='audio/wav')

    # Preprocess the audio file
    mfcc, signal, sample_rate = process_input(uploaded_file, 30)

    # # Display Waveform
    # st.subheader("Waveform")
    # fig, ax = plt.subplots()
    # librosa.display.waveshow(signal, sr=sample_rate, ax=ax)
    # st.pyplot(fig)

    # # Display Spectrogram
    # st.subheader("Spectrogram")
    # fig, ax = plt.subplots()
    # D = librosa.amplitude_to_db(np.abs(librosa.stft(signal)), ref=np.max)
    # librosa.display.specshow(D, sr=sample_rate, y_axis='log', x_axis='time', ax=ax)
    # st.pyplot(fig)

    # # Display MFCCs
    # st.subheader("Mel-frequency Cepstral Coefficients (MFCCs)")
    # fig, ax = plt.subplots()
    # librosa.display.specshow(mfcc.T, sr=22050, x_axis='time', ax=ax)
    # st.pyplot(fig)

    # Reshape MFCCs for model input
    X_to_predict = mfcc[np.newaxis, ..., np.newaxis]

    # Predict the genre
    prediction = model.predict(X_to_predict)
    predicted_index = np.argmax(prediction, axis=1)
    predicted_genre = genre_dict[int(predicted_index)]

    # Display the prediction
    st.subheader("Predicted Genre")
    st.write(f"The predicted genre is: **{predicted_genre}**")

    # Display genre probabilities
    st.subheader("Genre Probabilities")
    prob_df = pd.DataFrame(prediction[0], index=genre_dict.values(), columns=['Probability'])
    chart = alt.Chart(prob_df.reset_index()).mark_bar().encode(
        x='index:N',
        y='Probability:Q',
        color='index:N',
        tooltip=['index', 'Probability']
    ).interactive()
    st.altair_chart(chart, use_container_width=True)