import cv2
import numpy as np
# 1------------------------------------------------------- Detection of land and water---------------------------------------------------------
image = cv2.imread("4.png")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# HSV threshold
lowland = np.array([40, 80, 80])
uppland = np.array([80, 220, 220])
land_mask = cv2.inRange(hsv, lowland, uppland)

# Area filtering
contours, _ = cv2.findContours(land_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
clean_mask = np.zeros_like(land_mask)

for cnt in contours:
    if cv2.contourArea(cnt) > 5000:
        cv2.drawContours(clean_mask, [cnt], -1, 255, -1)

land_mask = clean_mask

# Visualization
land_visual = image.copy()
land_visual[land_mask == 255] = [0, 0, 0]


cv2.imshow("landonly",land_visual)
cv2.waitKey(0)
