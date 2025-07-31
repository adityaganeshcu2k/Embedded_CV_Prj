import cv2
import numpy as np
import itertools
import math
import time

frame_times = []
frame_index = 0

# Set this to 0 for webcam, or to a filename (e.g., 'rubiks_video.mp4')
video_source = 'sample.mp4'
cap = cv2.VideoCapture(video_source)

# Color thresholds (only white enabled; others can be uncommented)
color_ranges = {
    'white':  ([0, 0, 200],      [180, 50, 255],    (255, 255, 255)),  # low saturation, high value
    'yellow': ([20, 100, 100],   [35, 255, 255],    (0, 255, 255)),
    'orange': ([10, 100, 100],   [20, 255, 255],    (0, 165, 255)),
    'blue':   ([90, 100, 100],   [130, 255, 255],   (255, 0, 0)),
    'green':  ([40, 70, 100],    [85, 255, 255],    (0, 255, 0)),
    'red1':   ([0, 100, 100],    [10, 255, 255],    (0, 0, 255)),      # red lower hue
    'red2':   ([160, 100, 100],  [180, 255, 255],   (0, 0, 255)),      # red upper hue
}

def midpoint(p1, p2):
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

def is_parallelogram(pts, tol=5):
    pts = np.array(pts, dtype=np.float32)
    if len(pts) != 4:
        return False

    # Order points clockwise or counter-clockwise using convex hull
    hull = cv2.convexHull(pts).reshape(-1, 2)
    if len(hull) != 4:
        return False

    # Midpoints of diagonals
    mid1 = midpoint(hull[0], hull[2])
    mid2 = midpoint(hull[1], hull[3])

    dist = np.linalg.norm(np.array(mid1) - np.array(mid2))
    return dist < tol

def is_square(pts, tol=0.2):
    pts = np.array(pts, dtype=np.float32)
    hull = cv2.convexHull(pts).reshape(-1, 2)
    if len(hull) != 4:
        return False

    # Calculate all 6 distances between points
    dists = []
    for i in range(4):
        for j in range(i+1, 4):
            dists.append(np.linalg.norm(hull[i] - hull[j]))
    dists.sort()

    side = dists[0]
    diag = dists[-1]

    # Check four sides equal and two diagonals equal
    sides_equal = np.allclose(dists[0:4], side, rtol=tol)
    diags_equal = np.allclose(dists[4:6], diag, rtol=tol)
    # For square, diagonals must be longer than sides
    return sides_equal and diags_equal and diag > side

def draw_quad(img, pts, color=(0, 0, 255), thickness=3):
    pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)

def triangle_area(x1, y1, x2, y2, x3, y3):
    return 0.5 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

def find_by_index(centers, idx):
    for c in centers:
        if c['index'] == idx:
            return c
    return None
def get_average_color(lower, upper):
    return np.mean([lower, upper], axis=0)

def classify_color(hsv_avg):
    min_dist = float('inf')
    identified_color = "unknown"

    for name, (lower, upper, bgr) in color_ranges.items():
        lower = np.array(lower, dtype=np.float32)
        upper = np.array(upper, dtype=np.float32)
        reference = get_average_color(lower, upper)

        dist = np.linalg.norm(np.array(hsv_avg[:3]) - reference)
        if dist < min_dist:
            min_dist = dist
            identified_color = name

    return identified_color

def dist(p1, p2):
    return math.hypot(p2['cx'] - p1['cx'], p2['cy'] - p1['cy'])

def is_duplicate(cx, cy, seen, thresh=10):
    for _, sx, sy in seen:
        if abs(cx - sx) < thresh and abs(cy - sy) < thresh:
            return True
    return False

def triangle_area(x1, y1, x2, y2, x3, y3):
    return 0.5 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

while cap.isOpened():

    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive thresholding with larger blockSize and fine-tuned C value
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=21,  # Try values: 11, 15, 21
        C=4            # Increase this if too much is white
    )

    #edges = cv2.Canny(thresh, 50, 200)
    kernel = np.ones((2, 2), np.uint8)
    dilated_edges = cv2.dilate(thresh, kernel, iterations=4)
    # Step 2: Close gaps inside sticker areas
    closed = cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    output = frame.copy()
    seen_contours = []
    centers = []
    index = 1

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000 or area > 7000:
            continue

        epsilon = 0.05 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) > 8:
            continue

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [approx], -1, 255, -1)  # Filled contour

        mean_hsv = cv2.mean(hsv_frame, mask=mask)  # Returns (H, S, V, A)

        # Classify color
        color_name = classify_color(mean_hsv)

        M = cv2.moments(approx)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        if is_duplicate(cx, cy, seen_contours):
            continue

        seen_contours.append((area, cx, cy))
        centers.append({ 'index': index, 'cx': cx, 'cy': cy })

        # Determine BGR color from color_name
        bgr_color = (0, 0, 0)  # default to black
        if color_name in color_ranges:
            bgr_color = color_ranges[color_name][2]

        if frame_index == 463:
            if index == 13:
                p1 = np.array([cx, cy])
                distance = np.linalg.norm(p3 - p1)
                print("Distance between 15 and 22 point",distance)
            if index == 21:
                p2 = np.array([cx, cy])
                distance = np.linalg.norm(p1 - p2)
                print("Distance between 27 and 22 point",distance)
            if index == 4:
                p3 = np.array([cx, cy])

        # Draw center, text, and colored contour
        cv2.circle(output, (cx, cy), 3, (0, 0, 0), -1)
        cv2.putText(output, f"{index}", (cx - 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.drawContours(output, [approx], -1, bgr_color, 2)  # ← colored border
        index += 1
        
    center_points = [(c['cx'], c['cy']) for c in centers]

    # Save frames for debugging
    #cv2.imwrite(f'gray_{frame_index:04d}.jpg', gray)
    #cv2.imwrite(f'edges_{frame_index:04d}.jpg', edges)
    cv2.imwrite(f'dilated_{frame_index:04d}.jpg', closed)
    cv2.imwrite(f'output_{frame_index:04d}.jpg', output)

    # Show results
    cv2.imshow("Edges", closed)
    cv2.imshow("Detected Squares", output)

    end_time = time.time()
    frame_times.append(end_time - start_time)
    frame_index += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Print average FPS
if frame_times:
    avg_time_per_frame = sum(frame_times) / len(frame_times)
    avg_fps = 1.0 / avg_time_per_frame
    print(f"\nAverage FPS: {avg_fps:.2f}")
else:
    print("No frames processed.")
