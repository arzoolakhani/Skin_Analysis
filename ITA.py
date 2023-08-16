from derm_ita import get_ita
from PIL import Image
from derm_ita import get_kinyanjui_type
import csv
import os  
import cv2
import numpy as np 
"""
cam = cv2.VideoCapture(0)
harcascadePath = "haarcascade_frontalface_default.xml"
detector = cv2.CascadeClassifier(harcascadePath)

sampleNum = 0
while True:
    ret, img = cam.read()
    faces = detector.detectMultiScale(img, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)
        sampleNum += 1          
        cv2.imwrite(f"PBL{sampleNum}.jpg", img)
        cv2.imshow('frame',img)
    if cv2.waitKey(100) & 0xFF == ord('e'):
        break
    elif sampleNum > 5:
        break

cam.release()
cv2.destroyAllWindows()"""

for i in range(1, 7):
    image_path = f'PBL{i}.jpg'
    image = cv2.imread(image_path)

    whole_image_ita = get_ita(image=Image.fromarray(image))
    kinyanjui_type = get_kinyanjui_type(whole_image_ita)
    print(f'The ITA for image {i} is:', whole_image_ita)
    print(f'The Kinyanjui type for image {i} is:', kinyanjui_type)

with open('derm_data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image', 'ITA', 'Kinyanjui Type'])
    for i in range(1, 7):
        image_path = f'PBL{i}.jpg'
        whole_image_ita = get_ita(image=Image.open(image_path))
        kinyanjui_type = get_kinyanjui_type(whole_image_ita)
        writer.writerow([image_path, whole_image_ita, kinyanjui_type])
        print(f'Image {i} processed and added to CSV file.')