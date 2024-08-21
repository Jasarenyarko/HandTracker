import cv2
import time 
import mediapipe as mp


class handDector():
    def __init__(self,mode=False ,maxHands=2,model_complexity=1,detectionConfidence=0.5,trackingConfidence=0.5):
        self.mode = mode
        self.maxHands=maxHands
        self.model_complexity=model_complexity
        self.detectionConfidence = detectionConfidence
        self.trackingConfidence = trackingConfidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.model_complexity,
                                        self.detectionConfidence,self.trackingConfidence)
        self.mpDraw = mp.solutions.drawing_utils

    def findHand(self,img,draw=True):
        self.img = cv2.flip (img,1)
        imgRGB = cv2.cvtColor(self.img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(self.img,handLms,self.mpHands.HAND_CONNECTIONS)
         
        return self.img  
    
    def findPosition(self,img,draw=True):
        lmlist = []

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                for id, landmark in enumerate(handLms.landmark):
                    height,width,channel = self.img.shape
                    cx,cy = int(landmark.x*width),int(landmark.y*height)
                    lmlist.append([id,cx,cy])
                    if draw:
                        cv2.circle(img,[cx,cy],10,(0,0,255),cv2.FILLED)

        return lmlist


 


def main():
    cap=cv2.VideoCapture(0)
    preTime = 0
    currTime = 0
    detector = handDector()

    while True:
        success, img = cap.read()
        img = detector.findHand(img)
        lmlist = detector.findPosition(img)

        if len(lmlist)!= 0:
            print (lmlist[4])

        currTime = time.time()
        fps = 1/(currTime-preTime)
        preTime = currTime

        cv2.putText(img,str(int(fps)),(10,70),(cv2.FONT_HERSHEY_SIMPLEX),2,(0,0,0),4)

    
        if not success:
            break

        cv2.imshow ("Flipped Image",img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()        

if __name__ == "__main__":
    main() 