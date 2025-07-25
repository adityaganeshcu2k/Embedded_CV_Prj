import cv2
import numpy as np
import itertools
import math
import time

frame_times = []
frame_index = 0;
# Set this to 0 for webcam, or to a filename (e.g., 'rubiks_video.mp4')
video_source = 'sample.mp4'
cap = cv2.VideoCapture(video_source)

# Color thresholds (only white enabled; others can be uncommented)
color_ranges = {
    'white':  ([0, 0, 175],    [180, 40, 255], (255, 255, 255)),
    'yellow': ([26, 100, 200], [45, 255, 255], (0, 255, 255)),
    # 'orange': ([5, 100, 150], [10, 255, 255], (0, 165, 255)),
    'blue':   ([100, 140, 150], [140, 255, 255], (255, 0, 0)),
    'green':  ([50, 100, 120], [85, 230, 255], (0, 255, 0)),
    # 'red1':   ([0, 20, 100],   [5, 50, 255], (0, 0, 255)),
    # 'red2':   ([160, 100, 100], [180, 255, 255], (0, 0, 255)),
}

# Define sharpening kernel (same as your C code)
K = 0.001
kernel = np.array([
    [-K/8.0, -K/8.0, -K/8.0],
    [-K/8.0,  K+1.0, -K/8.0],
    [-K/8.0, -K/8.0, -K/8.0]
], dtype=np.float32)

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

    start_time = time.time()  # Start timer
    ret, frame = cap.read()
    if not ret:
        break

    #Uncomment the below lines to see the sharpened image in my opinion this does not work that good
    #alpha = 0.3  # Sharpening strength (0.0 = original, 1.0 = full sharpen)
    #sharpened = cv2.filter2D(frame, -1, kernel)
    #sharp_output = cv2.addWeighted(frame, 1 - alpha, sharpened, alpha, 0)
    #cv2.imshow("Sharpened", sharp_output)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Split HSV channels
    h, s, v = cv2.split(hsv)

    # Equalize the V channel
    v_equalized = cv2.equalizeHist(v)

    # Merge back the channels
    hsv_normalized = cv2.merge([h, s, v_equalized])
    frame_normalized = cv2.cvtColor(hsv_normalized, cv2.COLOR_HSV2BGR)
    output = frame.copy()
    index = 1
    seen_contours = []
    centers = []
    cnt_colors = { 'green': 0, 'white': 0, 'yellow': 0, 'orange': 0, 'blue': 0, 'red': 0 }

    cv2.imshow("HSV", frame_normalized)
    #cv2.imshow("Original", frame)

    # Detect colored blobs
    for color, (lower, upper, bgr) in color_ranges.items():
        lower_np = np.array(lower, dtype=np.uint8)
        upper_np = np.array(upper, dtype=np.uint8)

        mask = cv2.inRange(hsv_normalized, lower_np, upper_np)
        edges = cv2.Canny(mask, 5, 90)
        kernel = np.ones((2, 2), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            #print("Area Frame Indx",area,frame_index)
            if area < 2000:
                continue

            epsilon = 0.02 * cv2.arcLength(cnt, True)  # Adjust epsilon as needed
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            if len(approx) > 8:
                continue  # Keep only quadrilaterals (e.g., Rubik's cube facelets)

            M = cv2.moments(approx)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            if is_duplicate(cx, cy, seen_contours):
                continue

            seen_contours.append((area, cx, cy))
            centers.append({ 'index': index, 'cx': cx, 'cy': cy })

            if frame_index == 203:
            
                # === Get average HSV inside this contour ===
                mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [approx], -1, 255, -1)  # Filled contour on mask

                mean_hsv = cv2.mean(hsv, mask=mask)  # Returns (H, S, V, alpha)
                avg_h, avg_s, avg_v = mean_hsv[:3]
                print("Area Frame Indx",area,frame_index)
                print(f"Contour {index}: Avg HSV = ({avg_h:.2f}, {avg_s:.2f}, {avg_v:.2f})")
            

            # Draw center
            cv2.circle(output, (cx, cy), 3, (0, 0, 0), -1)
            cv2.putText(output, str(index), (cx - 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.drawContours(output, [approx], -1, bgr, 2)
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
                #cv2.line(output, (start['cx'], start['cy']), (end['cx'], end['cy']), (0, 0, 255), 1)

    end_time = time.time()  # End timer
    frame_times.append(end_time - start_time)
    # Display result
    cv2.imwrite(f'input_{frame_index:04d}.jpg', frame)
    cv2.imwrite(f'hsv_eq_{frame_index:04d}.jpg', frame_normalized)
    cv2.imwrite(f'hsv_{frame_index:04d}.jpg', mask)
    cv2.imwrite(f'canny_{frame_index:04d}.jpg', edges)
    cv2.imwrite(f'dilated_{frame_index:04d}.jpg', dilated_edges)
    cv2.imwrite(f'frame_{frame_index:04d}.jpg', output)
    frame_index += 1

    cv2.imshow("Rubik's Cube edge", dilated_edges)
    cv2.imshow("Rubik's Cube Tracking", output)

    end_time = time.time()  # End timer
    frame_times.append(end_time - start_time)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if frame_times:
    avg_time_per_frame = sum(frame_times) / len(frame_times)
    avg_fps = 1.0 / avg_time_per_frame
    print(f"\nAverage FPS: {avg_fps:.2f}")
else:
    print("No frames processed.")



