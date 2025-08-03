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

def find_by_index(centers, idx):
    for c in centers:
        if c['index'] == idx:
            return c
    return None

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

    AREA_TOLERANCE = 700

    if frame_index == 463:
        for p, q, r in itertools.combinations(range(1, len(centers) + 1), 3):
            p1 = find_by_index(centers, p)
            p2 = find_by_index(centers, q)
            p3 = find_by_index(centers, r)
            
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

                    if p1 == end:
                        start_indx = p
                        p6 = p1
                    elif p2 == end:
                        start_indx = q
                        p6 = p2
                    elif p3 == end:
                        start_indx = r
                        p6 = p3


                    
                    indices = [i for i in range(1, len(centers) + 1) if i != start_indx]
                    
                    for i, j in itertools.combinations(indices , 2):
                        
                        p4 = find_by_index(centers, i)
                        p5 = find_by_index(centers, j)


                        if p6 and p4 and p5:
                            area = triangle_area(p6['cx'], p6['cy'], p4['cx'], p4['cy'], p5['cx'], p5['cy'])
                            if area < AREA_TOLERANCE:
                                d14 = dist(p6, p4)
                                d45 = dist(p4, p5)
                                d15 = dist(p6, p5)

                                pairs = [(d14, (p6, p4, p5)),
                                         (d45, (p4, p5, p6)),
                                         (d15, (p6, p5, p4))]

                                longest, (start_next, end_next, middle_next) = max(pairs, key=lambda x: x[0])
 

                                if p1 == start_next:
                                    role = 'start'
                                elif p1 == end_next:
                                    role = 'end'
                                elif p1 == middle_next:
                                    role = 'middle'
                                else:
                                    role = 'not_in_triplet'  # shouldn't happen unless p1 is None
                                #print("p4 is:", role)

                                

                                # Make sure p6 == end
                                if not (p6['cx'] == end['cx'] and p6['cy'] == end['cy']):
                                    continue  # skip if not sharing vertex

                                # Pick a point from the second triplet that is NOT the shared one
                                for candidate in [p4, p5]:
                                    if candidate['cx'] != end['cx'] or candidate['cy'] != end['cy']:
                                        other = candidate
                                        break
                                else:
                                    continue  # no valid point to form the second vector

                                # Now compute vectors from the shared point `end`
                                v1 = (end['cx'] - start['cx'], end['cy'] - start['cy'])
                                v2 = (other['cx'] - end['cx'], other['cy'] - end['cy'])



                                dot = v1[0]*v2[0] + v1[1]*v2[1]
                                mag1 = math.hypot(*v1)
                                mag2 = math.hypot(*v2)

                                if mag1 != 0 and mag2 != 0:
                                    cos_theta = dot / (mag1 * mag2)
                                    angle_rad = math.acos(max(-1, min(1, cos_theta)))  # Clamp to avoid math domain errors
                                    angle_deg = math.degrees(angle_rad)

                                    # Determine if it's acute or obtuse
                                    nature = "Acute" if angle_deg < 90 else "Obtuse"

                                    
                                    if p1 == start:
                                        p7 = p1
                                    elif p2 == start:
                                        p7 = p2
                                    elif p3 == start:
                                        p7 = p3
                                    d1 = np.array([p6['cx'], p6['cy']])
                                    d2 = np.array([p7['cx'], p7['cy']])
                                    distance1 = np.linalg.norm(d1 - d2)

                                    if p4 == end_next:
                                        p8 = p4
                                    elif p5 == end_next:
                                        p8 = p5
                                    elif p4 == start_next:
                                        p8 = p4
                                    elif p5 == start_next:
                                        p8 = p5
                                    
                                    d3 = np.array([p6['cx'], p6['cy']])
                                    d4 = np.array([p8['cx'], p8['cy']])
                                    distance2 = np.linalg.norm(d3 - d4)
                                    
                                    if (60 <= angle_deg <= 120):
                                        if distance1 > distance2:
                                            if distance1 <= 1.5*distance2:
                                                # Draw line between collinear points
                                                cv2.line(output, (start['cx'], start['cy']), (end['cx'], end['cy']), (0, 0, 255), 1)
                                                cv2.line(output, (start_next['cx'], start_next['cy']), (end_next['cx'], end_next['cy']), (0, 255, 0), 1)
                                                print("p1 indx",p)
                                                print("p2 indx",q)
                                                print("p3 indx",r)
                                                print("p4 indx",i)
                                                print("p5 indx",j)
                                                
                                                
                                                print("Distance between p6 and p7",distance1)
                                                print("Distance between p6 and p8",distance2)
                                                print(f"Angle between lines: {angle_deg:.2f} degrees ({nature})")
                                        else:
                                            if distance2 <= 1.5*distance1:
                                                # Draw line between collinear points
                                                cv2.line(output, (start['cx'], start['cy']), (end['cx'], end['cy']), (0, 0, 255), 1)
                                                cv2.line(output, (start_next['cx'], start_next['cy']), (end_next['cx'], end_next['cy']), (0, 255, 0), 1)
                                                print("p1 indx",p)
                                                print("p2 indx",q)
                                                print("p3 indx",r)
                                                print("p4 indx",i)
                                                print("p5 indx",j)
                                                
                                                
                                                print("Distance between p6 and p7",distance1)
                                                print("Distance between p6 and p8",distance2)
                                                print(f"Angle between lines: {angle_deg:.2f} degrees ({nature})")
                                                

                                                                    

                
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
