from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Load faces_data
with open('data/faces_data.pkl', 'rb') as f:
    faces_data = pickle.load(f)

# Initialize variables for face recognition
confidence_levels = []
distances = []
accuracy_rates = []

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Simulate face recognition and populate confidence_levels, distances, and accuracy_rates
for _ in range(50):
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        confidence = np.random.uniform(80, 100)  # Replace with actual confidence value
        distance = np.random.uniform(30, 150)  # Replace with actual distance value
        accuracy = 100 - abs(90 - confidence)  # Adjust this calculation based on your requirements
        confidence_levels.append(confidence)
        distances.append(distance)
        accuracy_rates.append(accuracy)

# Visualize the distribution of distances vs confidence levels with accuracy as color
sns.set_style('darkgrid')
scatter = plt.scatter(confidence_levels, distances, c=accuracy_rates, cmap='coolwarm', marker='o', label='Face Recognition Results')
plt.title('Face Recognition Performance')
plt.xlabel('Confidence Level')
plt.ylabel('Distance (cms)')
plt.legend()

# Add colorbar for accuracy rates
cbar = plt.colorbar(scatter, label='Accuracy Rate')
cbar.set_label('Accuracy Rate', rotation=270, labelpad=15)

plt.show()

# Clean up
video.release()
cv2.destroyAllWindows()
