import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define the list of logo names and create a dictionary to map labels to numbers
disease_names = ['Acne', 'Actinic Keratosis', 'Basal Cell Carcinoma', 'Eczema', 'Rosacea']
label_dict = {disease_names[i]: i for i in range(len(disease_names))}

# Load the trained model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='linear', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='linear'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='linear'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='linear'))
model.add(Dropout(0.5))
model.add(Dense(len(disease_names), activation='softmax'))
# Save the model as an .h5 file
model.save('model_weights.h5')

