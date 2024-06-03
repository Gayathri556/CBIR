# offline_code.py

from PIL import Image
from feature_extractor1 import FeatureExtractor
from pathlib import Path
import numpy as np

if __name__ == '__main__':
    fe = FeatureExtractor()
    for img_path in sorted(Path("./static/img").glob("*.jpg")):
        try:
            img = Image.open(img_path)
            feature = fe.extract(img)
            feature_path = Path("./static/feature") / (img_path.stem + ".npy")
            np.save(feature_path, feature)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
