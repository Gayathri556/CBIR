# feature_extractor1.py

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import numpy as np
import speech_recognition as sr
import spacy

class FeatureExtractor:
    def __init__(self):
        self.base_model = VGG16(weights="imagenet")
        self.model = Model(inputs=self.base_model.input, outputs=self.base_model.get_layer("fc2").output)
        self.nlp = spacy.load("en_core_web_md")

    def extract(self, img):
        try:
            img = img.resize((224, 224)).convert("RGB")
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            feature = self.model.predict(x)[0]
            normalized_feature = feature / np.linalg.norm(feature)
            return normalized_feature
        except Exception as e:
            print(f"Error during image feature extraction: {e}")
            return np.array([])

    def extract_text(self, text):
        try:
            doc = self.nlp(text)
            text_feature = np.mean([word.vector for word in doc if word.has_vector], axis=0)
            if np.isnan(text_feature).any():
                return np.zeros(300)  # Or any other appropriate placeholder
            return text_feature
        except Exception as e:
            print(f"Error during text feature extraction: {e}")
            return np.zeros(300)

    def extract_voice(self, voice_query):
        try:
            recognizer = sr.Recognizer()
            with sr.AudioFile(voice_query) as source:
                audio_data = recognizer.record(source)
            text_query = recognizer.recognize_google(audio_data)
            voice_feature = self.extract_text(text_query)
            return voice_feature
        except Exception as e:
            print(f"Error during voice feature extraction: {e}")
            return np.zeros(300)  # Or any other appropriate placeholder
