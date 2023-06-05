import cv2
cap = cv2.VideoCapture('Walking.avi')
body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

while True:
    ret, frame = cap.read()
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bodies = body_cascade.detectMultiScale(grayframe, 1.2, 3)
    for (x,y,w,h) in bodies:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
    cv2.imshow("Detection of Video", frame)
    
    if cv2.waitKey(1) == 32:
        break

cap.release()
cv2.destroyAllWindows()
