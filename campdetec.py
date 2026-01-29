#original idea was to detect camps directly using hsv masks . it failed 
#reasons:
#1. grey camps had less satration 
#2. white had less satuiration as well so cant distinguish them.
#so i thought why not instead of fighting for grey-white ,why dont i accept grey as it is and detect camps differenly.
#thats when i thought color shouldnt define a camp
#its the shape that decides a camp .
#all camps are circular only then color is needed to further differentiate the camps(blue/pink/grey)
import cv2
import numpy as np
image = cv2.imread("4.png")
output=image.copy()
grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)#GREY for shape detection
hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)#HSV for color detection
blur = cv2.GaussianBlur(grey, (7, 7), 0)
edges = cv2.Canny(blur, 50, 150)
#here canny detecs whwre intensity changes fast.
#it keeps only the strongest edge line and removes thick edges
#50 is minimum gradient to be considered as a possible edge
#150 is a strong edge . above 150 is also a strong edge
#<50=discard
#50-150= weak edge or a possible edge
#>150= strong edge
contours, _ = cv2.findContours(
    edges,
    cv2.RETR_EXTERNAL,#gives external shapes
    cv2.CHAIN_APPROX_SIMPLE
)#findcontours return two values one contours= the list of boundary of white spaces we care about and other is hierarchy which is the info about parent/child contour relationships
#hierarchy helps dtect holes or distiguish inner vs outer boundaries. in camp detection we dont need this.

camps=[]
for cnt in contours:
    area= cv2.contourArea(cnt)
    if area < 1500 or area > 20000:
        continue

    perimeter = cv2.arcLength(cnt, True)#true means contour is closed
    if perimeter == 0:
        continue

    # Circularity check (circle ≈ 1.0)
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    if circularity < 0.6:
        continue

    M = cv2.moments(cnt)
    if M["m00"] == 0:
        continue

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    mask = np.zeros(grey.shape, dtype=np.uint8) #gray.shape means hight and width as same as that of grey which we generated above.np.zeroes creates an array full of 0. dtype.uint8 means pixel values can go from 0 to 255
    cv2.drawContours(mask, [cnt], -1, 255, -1)# here first -1 means select all contours in a list. [cnt] means only one contour is selected . 255 means color contour white . -1 this -1 means color it entirely
    meanhsv = cv2.mean(hsv, mask=mask)
    h=meanhsv[0]
    s=meanhsv[1]
    v=meanhsv[2]

    if s < 40:
        color = "Grey"
        capacity = 2
    elif 90 <= h <= 140:
        color = "Blue"
        capacity = 4
    elif 140 <= h <= 175:
        color = "Pink"
        capacity = 3
    else:
        continue

    camps.append({
        "location": (cx, cy),
        "color": color,
        "capacity": capacity
    })
    cv2.circle(output,(cx,cy),3,(0,0,255),-1)
    # here ouput is the image on which we are drawing
    #4 is radious of the circle which denotes the centre
# 0,0,255 is color of centre
#-1 means fill the centre completely red.
cv2.imshow("grey",grey)#grey helps us better identify shapes
cv2.imshow("final", output)
cv2.waitKey(0)
cv2.destroyAllWindows()