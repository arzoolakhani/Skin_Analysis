import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define the list of logo names and create a dictionary to map labels to numbers
disease_names = ['Acne and Rosacea Photos', 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions', 'Atopic Dermatitis Photos', 'Bullous Disease Photos', 'Eczema Photos','Melanoma Skin Cancer Nevi and Moles', 'Nail Fungus and other Nail Disease', 'Systemic Disease', 'Urticaria Hives', 'Vascular Tumors']
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
model.save('skin_disease_model.h5')



model.load_weights('skin_disease_classification.h5')

# Capture video from the camera
cap = cv2.VideoCapture(0)

# Loop over frames from the video feed
while True:
    # Read a frame from the video feed
    ret, frame = cap.read()

    # Preprocess the frame
    image = cv2.resize(frame, (64, 64))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Use the model to predict the skin disease
    prediction = model.predict(image)
    predicted_disease = disease_names[np.argmax(prediction)]

    # Display the predicted disease on the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, predicted_disease, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Skin Disease Classification', frame)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
