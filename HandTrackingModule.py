import cv2
import time 
import mediapipe as mp
import math

class HandDector():
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

    def FindHand(self,img,draw=True):
        self.img = cv2.flip (img,1)
        imgRGB = cv2.cvtColor(self.img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(self.img,handLms,self.mpHands.HAND_CONNECTIONS)
         
        return self.img  
    
    def FindPosition(self,img,draw=True,DrawBox = True):
        self.lmlist = []
        xList = []
        yList = []
        boundingBox = []
        xMin,xMax,yMin,yMax = 0,0,0,0

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                for id, landmark in enumerate(handLms.landmark):
                    height,width,channel = self.img.shape
                    cx,cy = int(landmark.x*width),int(landmark.y*height)
                    xList.append(cx)
                    yList.append(cy)
                    self.lmlist.append([id,cx,cy])
                    if draw:
                        cv2.circle(img,[cx,cy],10,(0,0,255),cv2.FILLED)
                print (xList)
                xMin, xMax = min(xList),max(xList)
                yMin, yMax = min(yList),max(yList)
                boundingBox = xMin,yMin,xMax,yMax
        if DrawBox:
            cv2.rectangle(self.img,(xMin,yMin),(xMax,yMax),(0,0,255),2)
            # print(xMax,xMin,yMax,yMin)
                        
        return self.lmlist, boundingBox
    

    def Highlight(self,img,position=[],cirle= True):
        self.position = position
        
        for i in self.position:
            x,y = self.lmlist[i][1], self.lmlist[i][2]

            if cirle:
                cv2.circle(img,(x,y),10,(0,0,255),cv2.FILLED)

    def DrawLineBetween(self,img,points=(),centre=True):
        firstPoint = self.lmlist[self.position[points[0]]]
        secondPoint = self.lmlist[self.position[points[1]]]
        color_circle = (0,0,255)

        x1,y1 = firstPoint[1],firstPoint[2]
        x2,y2 = secondPoint[1],secondPoint[2]

        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),3)

        centrePoint = ((x1+x2)//2 , (y1+y2)//2)

        lenLine = math.hypot(x2-x1,y2-y1)

        def drawCenterCircle(img,centrePoint,color_circle):
            cv2.circle(img,centrePoint,10,color_circle,cv2.FILLED) 

        if centre:
            drawCenterCircle(img,centrePoint=centrePoint,color_circle=color_circle)

            if lenLine < 50:
                color_circle =  (0,255,0) 
                drawCenterCircle(img,centrePoint=centrePoint,color_circle=color_circle)
        
        return lenLine
       

