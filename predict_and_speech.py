import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from gtts import gTTS
import pygame

model = load_model("asl_cnn_model.h5")

classes = sorted(os.listdir("asl_alphabet_train"))

def predict_letter(image):
    img = cv2.resize(image, (64, 64))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img)
    return classes[np.argmax(preds)]

sentence = ""

cap = cv2.VideoCapture(0)

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    x1, y1 = 100, 100
    x2, y2 = 300, 300
    cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)

    roi = frame[y1:y2, x1:x2]
    letter = predict_letter(roi)
    cv2.putText(frame, f"Predicted: {letter}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("ASL Recognizer", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        sentence += letter
        print("Sentence so far:", sentence)

    if key == ord('s'):
        if len(sentence) > 0:
            tts = gTTS(sentence)
            tts.save("asl_sentence.mp3")
            print("Speaking:", sentence)

            # play audio
            pygame.mixer.init()
            pygame.mixer.music.load("asl_sentence.mp3")
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pass
            sentence = ""


    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()