# UAS Search and Rescue – Computer Vision Pipeline

This repository contains a Computer Vision based pipeline developed as part of the **UAS Recruitment Task**.  
The objective is to analyze aerial images and assist in rescue decision-making by detecting land, camps, and casualties, followed by priority-based rescue logic.

---

##  Problem Overview

Given aerial images captured by a UAV:
- Segment land and water regions
- Detect rescue camps and their capacities
- Detect casualties and assign priorities based on shape
- Assign casualties to camps using logical constraints
- Compute an image-level rescue priority metric


##  Pipeline Stages

### Stage 1 – Land vs Water Segmentation
- Performed using HSV color thresholding
- Green regions are identified as land
- This stage intentionally over-segments green areas

---

### Stage 2 – Camp Detection
- Camps are detected using **geometric constraints**
  - Area filtering
  - Circularity check
- Color is used only after geometric validation
- Camp capacity is assigned based on color:
  - Blue → 4 casualties
  - Pink → 3 casualties
  - Grey → 2 casualties

---

### Stage 3 – Casualty Detection
- Land regions are removed to isolate foreground objects
- Binary thresholding and morphological operations are applied
- Shape approximation is used for classification:
  - Square → Priority 1
  - Triangle → Priority 2
  - Star → Priority 3

---

### Stage 4 – Casualty to Camp Assignment
- Casualties are sorted by priority (highest first)
- Each casualty is assigned to the nearest available camp
- Camp capacity constraints are enforced
- A greedy, priority-first strategy is used

---

### **Stage 5 – Camp Priority Calculation**
- Priorities of casualties assigned to each camp are summed
- Camp priorities are stored in the order:
[Blue, Pink, Grey]

---

### **Stage 6 – Rescue Priority Ratio (Pr)**
- Image-level urgency is computed as:
Pr = (Total priority of all camps) / (Number of casualties)
- This allows fair comparison across multiple images

---

##Image Ranking**
- Images are ranked in descending order of `Pr`
- Higher `Pr` indicates higher rescue urgency

---

##  Technologies Used
- Python
- OpenCV
- NumPy
- Git & GitHub

---

# Assumptions & Limitations
- Color-based segmentation may include non-land green objects
- Shape detection may misclassify some casualties
- Distance-based assignment is greedy, not globally optimal

These limitations are documented as part of the learning process.

---



