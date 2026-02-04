import cv2
import numpy as np
cass=[]
image=cv2.imread("4.png")
output=image.copy()
grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
blur = cv2.GaussianBlur(grey, (7, 7), 0)
edges = cv2.Canny(blur, 50, 150)
contours, _ = cv2.findContours(
    edges,
    cv2.RETR_EXTERNAL,#gives external shapes
    cv2.CHAIN_APPROX_SIMPLE
)
for cnt in contours:
    area=cv2.contourArea(cnt)

    if area<100 or area >1500:
        continue
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    M=cv2.moments(cnt)
    if M["m00"]==0:
        continue

    cx=int(M["m10"]/M["m00"])
    cy=int(M["m01"]/M["m00"])
    is_convex=cv2.isContourConvex(approx)
    if len(approx) == 3:
        shape = "triangle"
        priority = 2
    elif len(approx) == 4:
        shape = "square"
        priority = 1
    elif len(approx)>4:
        shape = "star"
        priority = 3

    cass.append({
        "location": (cx, cy),
        "type": shape,
        "priority": priority
    })
#visual detection below
debug = image.copy()

for c in cass:
    cx, cy = c["location"]
    cv2.circle(debug, (cx, cy), 4, (0, 0, 255), -1)
    cv2.putText(
        debug,
        c["type"],
        (cx + 5, cy),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255,255,255),
        1
    )


cv2.imshow("casualty", debug)
cv2.waitKey(0)
cv2.destroyAllWindows()
