from flask import Flask, request, render_template
from pathlib import Path
from PIL import Image
from feature_extractor1 import FeatureExtractor
from text_processing import process_text_query
import numpy as np
import os
import speech_recognition as sr
from datetime import datetime

app = Flask(__name__)

app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'm4a'}

fe = FeatureExtractor()
features = []
img_paths = []

# Read image features
for feature_path in Path("./static/feature").glob("*.npy"):
    features.append(np.load(feature_path))
    img_paths.append(Path("./static/img") / (feature_path.stem + ".jpg"))

# Replace backslashes with forward slashes in image paths
img_paths = [str(path).replace(os.path.sep, '/') for path in img_paths]
features = np.array(features)


def convert_m4a_to_wav(m4a_file, output_folder):
    try:
        os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist
        
        # Construct output file path with a unique name based on the current timestamp
        wav_filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".wav"
        wav_file = os.path.join(output_folder, wav_filename)
        
        # Save the m4a file to a temporary location
        m4a_tempfile = os.path.join(output_folder, "temp.m4a")
        m4a_file.save(m4a_tempfile)
        
        # Use ffmpeg to convert m4a to wav
        command = f"ffmpeg -i {m4a_tempfile} {wav_file}"
        print("Executing command:", command)
        os.system(command)
        
        # Check if the WAV file was created
        if os.path.isfile(wav_file):
            print("Conversion successful. WAV file created:", wav_file)
            return wav_file
        else:
            print("Conversion failed. WAV file not created.")
            return None
    except Exception as e:
        print("Error converting m4a to wav:", e)
        return None



# Function to perform speech recognition
def speech_to_text_function(voice_file):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(current_directory, "checkfile")
    converted_wav_file = convert_m4a_to_wav(voice_file, output_folder)
    if not converted_wav_file:
        return "Error: Failed to convert m4a file to wav."

    print("Converted wav file path:", converted_wav_file)  # Debugging statement

    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # Load the audio file
    with sr.AudioFile(converted_wav_file) as source:
        audio_data = recognizer.record(source)

    # Recognize speech using Google Web Speech API
    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return "Could not understand the audio"
    except sr.RequestError as e:
        return "Error occurred during speech recognition: {0}".format(e)


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Remaining code remains the same...

def handle_text_query(text_query):
    print("Received text query:", text_query)
    img_paths = process_text_query(text_query)
    print("Matching image paths:", img_paths)
    if not img_paths:
        print("No matching images found for text query.")
        return render_template('index1.html', error_message="Error: No matching images found for the text query.")
    return render_template('index1.html', text_query=text_query, img_paths=img_paths)

def handle_image_query(file):
    global img_paths  # Declare img_paths as a global variable
    
    if file.filename == '':
        print("Error: No file selected.")
        return render_template('index1.html', error_message="Error: No file selected.")
    
    if file and allowed_file(file.filename):
        try:
            img = Image.open(file.stream)
            uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
            img.save(uploaded_img_path)
            print("Saved uploaded image to:", uploaded_img_path)
            query = fe.extract(img)
            
            # Perform nearest neighbor search
            dists = np.linalg.norm(features - query, axis=1)
            sorted_indices = np.argsort(dists)  # Sort indices based on distance
            scores = [(dists[idx], img_paths[idx]) for idx in sorted_indices if idx < len(img_paths) and dists[idx] <= 1.0]
            print("Found matching images for query:", scores)
            
            # Pass relevant variables to the template
            return render_template('index1.html', uploaded_img_path=uploaded_img_path, scores=scores)
        except Exception as e:
            print("Error opening uploaded image:", e)
            return render_template('index1.html', error_message="Error: Invalid image file.")
    else:
        print("Error: Unsupported file format.")
        return render_template('index1.html', error_message="Error: Unsupported file format.")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if text query is submitted
        if request.form.get('text_query'):
            text_query = request.form['text_query']
            return handle_text_query(text_query)
        
        # Check if image query is submitted
        elif request.files.get('query_img'):
            print("Received image query")
            print("Keys in request.files:", request.files.keys())
            return handle_image_query(request.files['query_img'])
        
        # Check if voice query is submitted
        elif 'voice_query' in request.files:
            print("Received voice query")
            voice_query = request.files['voice_query']
            text_query = speech_to_text_function(voice_query)
            return handle_text_query(text_query)

        else:
            print("Invalid input")
    
    # If no query is submitted or if it's a GET request, render the initial template
    return render_template('index1.html')


if __name__ == "__main__":
    app.run("0.0.0.0")