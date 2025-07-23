import cv2
import numpy as np
import itertools
import math

# Set this to 0 for webcam, or to a filename (e.g., 'rubiks_video.mp4')
video_source = 'sample.mp4'
cap = cv2.VideoCapture(video_source)

# Color thresholds (only white enabled; others can be uncommented)
color_ranges = {
    'white':  ([0, 0, 200],    [180, 40, 255], (255, 255, 255)),
    # 'yellow': ([26, 100, 100], [45, 255, 255], (0, 255, 255)),
    # 'orange': ([5, 100, 150], [10, 255, 255], (0, 165, 255)),
    # 'blue':   ([100, 100, 100], [140, 255, 255], (255, 0, 0)),
    # 'green':  ([50, 100, 100], [85, 255, 255], (0, 255, 0)),
    # 'red1':   ([0, 20, 100],   [5, 50, 255], (0, 0, 255)),
    # 'red2':   ([160, 100, 100], [180, 255, 255], (0, 0, 255)),
}

def triangle_area(x1, y1, x2, y2, x3, y3):
    return 0.5 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

def find_by_index(centers, idx):
    for c in centers:
        if c['index'] == idx:
            return c
    return None

def dist(p1, p2):
    return math.hypot(p2['cx'] - p1['cx'], p2['cy'] - p1['cy'])

def is_duplicate(cx, cy, seen, thresh=10):
    for _, sx, sy in seen:
        if abs(cx - sx) < thresh and abs(cy - sy) < thresh:
            return True
    return False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    output = frame.copy()
    index = 1
    seen_contours = []
    centers = []
    cnt_colors = { 'green': 0, 'white': 0, 'yellow': 0, 'orange': 0, 'blue': 0, 'red': 0 }

    # Detect colored blobs
    for color, (lower, upper, bgr) in color_ranges.items():
        lower_np = np.array(lower, dtype=np.uint8)
        upper_np = np.array(upper, dtype=np.uint8)

        mask = cv2.inRange(hsv, lower_np, upper_np)
        edges = cv2.Canny(mask, 5, 90)
        kernel = np.ones((2, 2), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(dilated_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            if is_duplicate(cx, cy, seen_contours):
                continue

            seen_contours.append((area, cx, cy))
            centers.append({ 'index': index, 'cx': cx, 'cy': cy })

            # Draw center
            cv2.circle(output, (cx, cy), 3, (0, 0, 0), -1)
            cv2.putText(output, str(index), (cx - 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.drawContours(output, [cnt], -1, bgr, 2)
            index += 1

            if 'red' in color:
                cnt_colors['red'] += 1
            else:
                cnt_colors[color] += 1

    # Triangle (collinear) analysis
    AREA_TOLERANCE = 700
    for i, j, k in itertools.combinations(range(1, len(centers) + 1), 3):
        p1 = find_by_index(centers, i)
        p2 = find_by_index(centers, j)
        p3 = find_by_index(centers, k)

        if p1 and p2 and p3:
            area = triangle_area(p1['cx'], p1['cy'], p2['cx'], p2['cy'], p3['cx'], p3['cy'])
            if area < AREA_TOLERANCE:
                d12 = dist(p1, p2)
                d23 = dist(p2, p3)
                d13 = dist(p1, p3)

                pairs = [(d12, (p1, p2, p3)),
                         (d23, (p2, p3, p1)),
                         (d13, (p1, p3, p2))]

                longest, (start, end, middle) = max(pairs, key=lambda x: x[0])

                # Draw line between collinear points
                cv2.line(output, (start['cx'], start['cy']), (end['cx'], end['cy']), (0, 0, 255), 1)

    # Display result
    cv2.imshow("Rubik's Cube Tracking", output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
