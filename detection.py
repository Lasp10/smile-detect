import cv2

smile_classifier = cv2.CascadeClassifier("haarcascade_smile.xml")

frontalface_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
smiling = False
while cap.isOpened():
    _, frame = cap.read()

    frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    

    face = frontalface_classifier.detectMultiScale(frame2, scaleFactor = 1.1, minNeighbors = 5)
    
    for (x, y, w, h) in face:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        face_regiongray = frame2[y:y+h, x:x+w]
        face_regioncolor = frame[y:y+h, x:x+w]
        smile = smile_classifier.detectMultiScale(face_regiongray, scaleFactor = 1.3, minNeighbors = 20)
        for (x2,y2,w2,h2) in smile:
            cv2.rectangle(face_regioncolor, (x2, y2), (x2+w2, y2+h2), (0, 0, 255), 7)
            smiling = True

            if smiling == True:
                cv2.putText(frame, 'Smiling', (x2, y2), cv2.FONT_ITALIC, 1, (255, 0, 0), 1)
        
        
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
