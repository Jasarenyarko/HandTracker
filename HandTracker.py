import cv2
import time 
import mediapipe as mp

cap=cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

preTime = 0
currTime = 0

while True:
    success, img = cap.read()
    flipped_img = cv2.flip (img,1)
    imgRGB = cv2.cvtColor(flipped_img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, landmark in enumerate(handLms.landmark):
                height,width,channel = flipped_img.shape
                cx,cy = int(landmark.x*width),int(landmark.y*height)


            mpDraw.draw_landmarks(flipped_img,handLms,mpHands.HAND_CONNECTIONS)

    
    currTime = time.time()
    fps = 1/(currTime-preTime)
    preTime = currTime

    cv2.putText(flipped_img,str(int(fps)),(10,70),(cv2.FONT_HERSHEY_SIMPLEX),2,(0,0,0),4)

    
    if not success:
        break

    cv2.imshow ("Flipped Image",flipped_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()