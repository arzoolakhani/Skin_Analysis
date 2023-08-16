import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define the list of skin diseases and create a dictionary to map labels to numbers
disease_names = ['Acne and Rosacea Photos', 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions', 'Atopic Dermatitis Photos', 'Bullous Disease Photos', 'Eczema Photos','Melanoma Skin Cancer Nevi and Moles', 'Nail Fungus and other Nail Disease', 'Systemic Disease', 'Urticaria Hives', 'Vascular Tumors']
label_dict = {disease_names[i]: i for i in range(len(disease_names))}

# Load the images and labels from the photos folder
images = []
labels = []
for disease_name in disease_names:
    folder_path = os.path.join('Dataset', disease_name)
    for filename in os.listdir(folder_path):         
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (64, 64))
        images.append(image)
        labels.append(label_dict[disease_name])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)

# Convert the data to arrays and normalize the pixel values
X_train = np.array(X_train) / 255.0
X_test = np.array(X_test) / 255.0
y_train = np.array(y_train)
y_test = np.array(y_test)

# Define the CNN model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(disease_names), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# Use the model to predict the logos in a new image
image_path = 'trial.jpg'
image = cv2.imread(image_path)
image = cv2.resize(image, (64, 64))
image = np.array(image) / 255.0
image = np.expand_dims(image, axis=0)
prediction = model.predict(image)
predicted_disease = disease_names[np.argmax(prediction)]
print('The detected skin disease is:', predicted_disease)
