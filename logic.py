import cv2
import numpy as np
import math
#load image
image = cv2.imread("4.png")
output = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

H, W = gray.shape



lowerland = np.array([35, 40, 40], dtype=np.uint8)
upperland = np.array([85, 255, 255], dtype=np.uint8)

landmask = cv2.inRange(hsv, lowerland, upperland)


camps = []
campmask = np.zeros((H, W), dtype=np.uint8)

blur = cv2.GaussianBlur(gray, (7, 7), 0)
edges = cv2.Canny(blur, 50, 150)

contours, _ = cv2.findContours(
    edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 1500 or area > 20000:
        continue

    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        continue

    circularity = 4 * np.pi * area / (perimeter * perimeter)
    if circularity < 0.65:
        continue

    M = cv2.moments(cnt)
    if M["m00"] == 0:
        continue

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)

    mean_hsv = cv2.mean(hsv, mask=mask)
    h, s, v = mean_hsv[:3]

    if s < 40:
        color, capacity = "Grey", 2
    elif 90 <= h <= 140:
        color, capacity = "Blue", 4
    elif 140 <= h <= 175:
        color, capacity = "Pink", 3
    else:
        continue

    camps.append({
        "color": color,
        "location": (cx, cy),
        "capacity": capacity
    })

    campmask[mask == 255] = 255

    cv2.circle(output, (cx, cy), 8, (0, 0, 255), -1)
    cv2.putText(
        output, f"{color}",
        (cx - 30, cy - 10),
        cv2.FONT_HERSHEY_TRIPLEX, 0.5,
        (255, 255, 255), 1
    )


casualties = []
casualty_mask = np.zeros((H, W), dtype=np.uint8)

# here i remove land to make sure that objects stand out
non_land = cv2.bitwise_not(landmask)

_, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
binary = cv2.bitwise_and(binary, non_land)


contours, _ = cv2.findContours(
    binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 150 or area > 2500:
        continue

    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    M = cv2.moments(cnt)
    if M["m00"] == 0:
        continue

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    if len(approx) == 3:
        priority, label = 2, "Triangle"
    elif len(approx) == 4:
        priority, label = 1, "Square"
    else:
        priority, label = 3, "Star"

    casualties.append({
        "location": (cx, cy),
        "priority": priority
    })

    cv2.drawContours(casualty_mask, [cnt], -1, 255, -1)

    cv2.circle(output, (cx, cy), 4, (255, 0, 0), -1)
    cv2.putText(
        output, label,
        (cx + 5, cy),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
        (255, 255, 255), 1
    )


def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

assignments = []

# High priority rescued first
casualties_sorted = sorted(
    casualties, key=lambda x: x["priority"], reverse=True
)

for casualty in casualties_sorted:

    available_camps = [c for c in camps if c["capacity"] > 0]
    if not available_camps:
        break

    nearest = min(
        available_camps,
        key=lambda c: distance(casualty["location"], c["location"])
    )

    assignments.append({
        "casualty": casualty,
        "camp": nearest
    })

    nearest["capacity"] -= 1



cv2.imshow("Stage 1-4 Output", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
