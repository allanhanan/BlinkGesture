import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

#path to the saved model
model_file_path = './BDmodel.keras'

#load model
model = tf.keras.models.load_model(model_file_path)

#preprocess image
def preprocess_image(img_path, target_size=(64, 64)):
    img = image.load_img(img_path, target_size=target_size)  #load image
    img_array = img_to_array(img)                            #convert image to a numpy array
    img_array = np.expand_dims(img_array, axis=0)            #add a batch dimension
    img_array = img_array / 255.0                           #normalize pixel values
    return img_array

#path to test image
image_path = './closed1.jpg'

#preprocess the image
preprocessed_image = preprocess_image(image_path)

#make predictions
predictions = model.predict(preprocessed_image)
open_eye_score = predictions[0][1]
closed_eye_score = predictions[0][0]

#results
if open_eye_score > 0.5 and closed_eye_score < 0.5:
    print("The model predicts: Open Eye with score", open_eye_score)
elif closed_eye_score > 0.5 and open_eye_score < 0.5:
    print("The model predicts: Closed Eye with score", closed_eye_score)
else:
    print("The model is uncertain or the eye is not present. Open Eye score:", open_eye_score, "Closed Eye score:", closed_eye_score)
