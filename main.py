import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load threshold values from .env file
SSIM_THRESHOLD = float(os.getenv("SSIM_THRESHOLD", 0.6))
FACE_SIZE = int(os.getenv("FACE_SIZE", 200))
BLUR_KERNEL = int(os.getenv("BLUR_KERNEL", 3))

# Load known faces from a folder (authorized users)
KNOWN_FACES_DIR = "known_faces"
known_faces = []
known_names = []

# Image formats allowed
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

for filename in os.listdir(KNOWN_FACES_DIR):
    img_path = os.path.join(KNOWN_FACES_DIR, filename)
    ext = os.path.splitext(filename)[1].lower()
    if ext in ALLOWED_EXTENSIONS:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (FACE_SIZE, FACE_SIZE))
        img = cv2.equalizeHist(img)
        known_faces.append(img)
        known_names.append(os.path.splitext(filename)[0]) 

# Function to recognize faces
def recognize_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (FACE_SIZE, FACE_SIZE))
        face = cv2.GaussianBlur(face, (BLUR_KERNEL, BLUR_KERNEL), 0)
        
        # Compare with known faces using SSIM
        max_similarity = -1  # SSIM ranges from -1 to 1
        identity = "Unknown"
        
        for i, known_face in enumerate(known_faces):
            similarity = ssim(face, known_face)
            if similarity > max_similarity:
                max_similarity = similarity
                identity = known_names[i]
        
        # Set threshold for recognition (SSIM closer to 1 is a better match)
        if max_similarity > SSIM_THRESHOLD:
            label = f"Access Granted: {identity}"
            color = (0, 255, 0)
        else:
            label = "Access Denied"
            color = (0, 0, 255)
        
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    
    return frame

# Open webcam for real-time face recognition
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = recognize_face(frame)
    cv2.imshow('Biometric Authentication', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

