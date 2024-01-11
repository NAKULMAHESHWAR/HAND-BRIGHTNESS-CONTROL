from platform import release
import mediapipe as mp
import cv2
import numpy as np
import screen_brightness_control as sbc
from math import hypot

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

while True:
    success, img = cap.read()
    if not success:
        break

imgRGB = cv2.cvtColor(img, cv2.COLOUR_BGR2RGB)
results = hands.process(imgRGB)
lmList = []
if results.multi_hand_landmarks:
    for handlandmark in results.multi_hand_landmarks:
        for id, lm in enumerate(handlandmark.landmark):
            h, w, _ = img.shape
            cx,cy = int(lm.x * w), int(lm.y *h)
            lmList.append([id, cx, cy])

            # Draw Landmarks
            mpDraw.drawlandmarks(img, handlandmark, mpHands.HAND_CONNECTIONS,
                                 landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                                 connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())
            
            #brightness control
            if lmList!=[]:
                x1,y1 = lmList[4][1], lmList[4][2]
                x2,y2 = lmList[8][1], lmList[8][2]
                cv2.circle(img, (x1,y1),4,(255,0,0), cv2.FILLED)
                cv2.circle(img, (x2,y2),4,(255,0,0), cv2.FILLED)

                #drawing line between thumb and index finger

                cv2.line(img, (x1,y1),(x2,y2), (255,0,0), 3)

                length = hypot(x2-x1, y2-y1)
                bright = np.interp(length, [15,240],[0,100])
                print(bright, length)
                sbc.set_brightness(int(bright))
                cv2.imshow('Image', img)
   



release()
cv2.destroyAllWindows()