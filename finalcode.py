import cv2
import numpy as np
import math
from collections import defaultdict

image_name = "4.png"
image = cv2.imread(image_name)
output = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

H, W = gray.shape

lower_land = np.array([35, 40, 40], dtype=np.uint8)
upper_land = np.array([85, 255, 255], dtype=np.uint8)

land_mask = cv2.inRange(hsv, lower_land, upper_land)

# STAGE 2: CAMP DETECTION 


camps = []
camp_mask = np.zeros((H, W), dtype=np.uint8)

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

    camp_mask[mask == 255] = 255

    cv2.circle(output, (cx, cy), 8, (0, 0, 255), -1)
    cv2.putText(
        output, f"{color}",
        (cx - 30, cy - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (255, 255, 255), 1
    )

# STAGE 3: CASUALTY DETECTION (FILLED SHAPES)


casualties = []
casualty_mask = np.zeros((H, W), dtype=np.uint8)

non_land = cv2.bitwise_not(land_mask)

_, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
binary = cv2.bitwise_and(binary, non_land)

kernel = np.ones((3, 3), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, 2)
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, 1)

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

# =========================================================
# STAGE 4: CASUALTY → CAMP ASSIGNMENT
# =========================================================

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

assignments = []

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


# STAGE 5: CAMP PRIORITY CALCULATION


camp_priority_map = defaultdict(int)
total_casualties = 0

for a in assignments:
    camp_priority_map[a["camp"]["color"]] += a["casualty"]["priority"]
    total_casualties += 1

camp_priority = [
    camp_priority_map.get("Blue", 0),
    camp_priority_map.get("Pink", 0),
    camp_priority_map.get("Grey", 0)
]


# STAGE 6: IMAGE RESCUE PRIORITY RATIO


Pr = sum(camp_priority) / total_casualties if total_casualties > 0 else 0

# STAGE 7: IMAGE RANKING (SINGLE IMAGE SHOWN FOR COMPLETENESS)


results = [{
    "image_name": image_name,
    "camp_priority": camp_priority,
    "Pr": Pr
}]

sorted_images = sorted(results, key=lambda x: x["Pr"], reverse=True)
image_by_rescue_ratio = [img["image_name"] for img in sorted_images]

# FINAL OUTPUT


print("Camp_priority =", camp_priority)
print("Priority_ratio (Pr) =", Pr)
print("image_by_rescue_ratio =", image_by_rescue_ratio)

cv2.imshow("Final Output (Stage 1–4 Visual)", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
