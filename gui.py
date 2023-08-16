import tkinter as tk
import cv2
import numpy as np
from PIL import Image
from derm_ita import get_ita
from derm_ita import get_kinyanjui_type
from keras.models import load_model
import csv

model = load_model('model_weights.h5')

disease_names = ['Acne', 'Actinic Keratosis', 'Basal Cell Carcinoma', 'Eczemaa', 'Rosacea']
label_dict = {disease_names[i]: i for i in range(len(disease_names))}

def take_images():
    cam = cv2.VideoCapture(0)
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)

    sampleNum = 0
    while True:
        ret, img = cam.read()

        # detect faces
        faces = detector.detectMultiScale(img, 1.3, 5)

        # use skin color detector to get region of skin
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin = cv2.bitwise_and(img, img, mask=mask)

        # use largest skin region as the ROI
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in contours]
        if areas:
            max_index = np.argmax(areas)
            (x, y, w, h) = cv2.boundingRect(contours[max_index])
            roi = skin[y:y+h, x:x+w]
        else:
            roi = img

        # draw rectangle around face
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)
        
        # write image to file
        sampleNum += 1
        cv2.imwrite(f"PBL{sampleNum}.jpg", roi)
        cv2.imshow('frame',img)

        # check for user input
        if cv2.waitKey(100) & 0xFF == ord('e'):
            break
        elif sampleNum > 4:
            break

    cam.release()
    cv2.destroyAllWindows()
    
def display_images():
    image_path = f'PBL1.jpg'
    image = cv2.imread(image_path)
    image = cv2.resize(image, (300,300))
    cv2.imshow(f"Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def calculate_ita_kinyanjui():
    # create the textbox
    textbox = tk.Text(root, height=10, width=50)
    textbox.pack(pady=10)

    for i in range(1, 7):
        image_path = f'PBL{i}.jpg'
        image = cv2.imread(image_path)

        whole_image_ita = get_ita(image=Image.fromarray(image))
        kinyanjui_type = get_kinyanjui_type(whole_image_ita)
        print(f'The ITA for image {i} is:', whole_image_ita)
        print(f'The Kinyanjui type for image {i} is:', kinyanjui_type)

        image_path1 = f'PBL{i}.jpg'
        image1 = cv2.imread(image_path1)
        image1 = cv2.resize(image1, (64, 64))
        image1 = np.array(image1) / 255.0
        image1 = np.expand_dims(image1, axis=0)
        prediction = model.predict(image1)
        predicted_idx = np.argmax(prediction)
        if prediction[0][predicted_idx] < 0.5:
            predicted = 'No disease detected'
        else:
            predicted = disease_names[predicted_idx]

        # insert the computed values into the textbox
        textbox.insert(tk.END, f'Image {i}:\nITA: {whole_image_ita}\nKinyanjui Type: {kinyanjui_type}\nPredicted: {predicted}\n\n')
        
    with open('derm_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image', 'ITA', 'Kinyanjui Type','Predicted'])
        for i in range(1, 7):
            image_path = f'PBL{i}.jpg'
            whole_image_ita = get_ita(image=Image.open(image_path))
            kinyanjui_type = get_kinyanjui_type(whole_image_ita)    
            writer.writerow([image_path, whole_image_ita, kinyanjui_type,predicted])
            print(f'Image {i} processed and added to CSV file.')



# create GUI
root = tk.Tk()
root.geometry('400x400')

# set background color
root.configure(bg='lightblue')

# create title label
title_label = tk.Label(root, text='Skin Disease and Tone', font=('Arial', 40), fg='white', bg='lightblue')

# create buttons

take_images_button = tk.Button(root, text="Take Image", fg="white"  ,bg="lightgreen"  ,width=20  ,height=2 ,activebackground = "white" ,font=('times', 15, ' bold '),command =take_images)
display_images_button = tk.Button(root, text="Display Images", fg="white"  ,bg="lightgreen"  ,width=20  ,height=2 ,activebackground = "white" ,font=('times', 15, ' bold '),command =display_images)
calculate_button = tk.Button(root, text="Calculate Skin Tone ", fg="white"  ,bg="lightgreen"  ,width=20  ,height=2 ,activebackground = "white" ,font=('times', 15, ' bold '),command =calculate_ita_kinyanjui)

# place buttons in the window
title_label.pack(pady=10)
take_images_button.pack(side=tk.LEFT, padx=10, pady=10)
display_images_button.pack(side=tk.LEFT, padx=10, pady=10)
calculate_button.pack(side=tk.LEFT, padx=10, pady=10)

root.mainloop()