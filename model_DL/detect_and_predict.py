import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model

# Load model
model = load_model("/Users/miriambenali/Desktop/Project-Simplon/Facial-Detection_DeepLearning/model_DL/facial_model.keras")

# Define emotion classes
classes = ['angry', 'disgusted', 'fearful', 'happy', 'neutral','sad','surprised']

# Load Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# load picture for test
image_path="//Users/miriambenali/Desktop/Project-Simplon/Facial-Detection_DeepLearning/model_DL/myimage/6.jpg"
image = cv2.imread(image_path)


if image is None:
    print("Error: Image not founded")
    exit()

def detect_and_predict_emotion(image):
    #convert image to gray
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

    #detector faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    print(faces)


   #Loop for each face detected
    for (x,y,h,w) in faces:
        cv2.rectangle(image, (x,y),(x+w,y+h),(0,255,0),2) #it makes a rectangle
        roi= gray[y:y+h, x:x+w] #extraire face
        roi_resized= cv2.resize(roi,(48,48)) #reshap take same shapes of model
        roi_reshaped= np.expand_dims(roi_resized, axis=(0,-1)) 
        roi_normalize= roi_reshaped/255.0 #divide on 255 to take numbers between 0 and 1


        prediction= model.predict(roi_normalize,verbose=0) #prediction for each emotion
        emotion_index= np.argmax(prediction) # it gives the best emotion predicted
        emotion_label= classes[emotion_index] #name of emotion
        confidence= prediction[0][emotion_index]

        #show rectangle on image
        text=f"{emotion_label} ({confidence:.2f})"
        cv2.putText(image,text, (x,y -10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,(0,255,0),2)
        
        return emotion_label, confidence
    
        ''' #show image
    cv2.imshow("Résultat - Détection d'émotions",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''

#print(detect_and_predict_emotion(image))


#########################
#test model on video live
'''
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No cam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Dessiner le rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 225, 0), 2)

        # Extraire le visage
        roi = gray[y:y + h, x:x + w]
        roi_resized = cv2.resize(roi, (48, 48))
        roi_reshaped = np.expand_dims(roi_resized, axis=(0, -1))

        # Prédiction
        prediction = model.predict(roi_reshaped, verbose=0)
        emotion_index = np.argmax(prediction)
        score = prediction[0][emotion_index]
        emotion_label = classes[emotion_index]

        # Afficher l'émotion sur l'image
        cv2.putText(frame, str(score), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Montrer la vidéo
    cv2.imshow("Détection d'émotions", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()'''