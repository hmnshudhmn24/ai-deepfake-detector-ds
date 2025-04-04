import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import argparse

def predict_deepfake(img_path, model_path="deepfake_detector_model.h5"):
    model = load_model(model_path)
    img = image.load_img(img_path, target_size=(160, 160))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    result = 'Fake' if prediction > 0.5 else 'Real'
    print(f'Prediction: {result} ({prediction*100:.1f}% confidence)')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help="Path to input image")
    args = parser.parse_args()
    predict_deepfake(args.image)
