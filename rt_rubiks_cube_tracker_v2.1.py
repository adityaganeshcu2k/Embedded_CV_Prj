import cv2
import numpy as np
import itertools
import math
import time

frame_times = []
frame_index = 0
AREA_TOLERANCE = 900
# Set this to 0 for webcam, or to a filename (e.g., 'rubiks_video.mp4')
video_source = 'sample6.mp4'
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

def angle_between(v1, v2):
    dot = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm_product == 0:
        return 0
    cos_theta = np.clip(dot / norm_product, -1.0, 1.0)
    return math.degrees(math.acos(cos_theta))

def find_all_next_collinear_sides(centers, fixed_point, area_tol=AREA_TOLERANCE, used_points=None, allow_point=None):
    if used_points is None:
        used_points = set()

    candidates = []
    for c in centers:
        if c == fixed_point:
            continue
        if c['index'] in used_points:
            if allow_point is not None and c == allow_point:
                candidates.append(c)
            else:
                continue
        else:
            candidates.append(c)

    results = []
    for p, q in itertools.combinations(candidates, 2):
        area = triangle_area(fixed_point['cx'], fixed_point['cy'],
                             p['cx'], p['cy'],
                             q['cx'], q['cy'])
        if area < area_tol:
            d1 = dist(fixed_point, p)
            d2 = dist(p, q)
            d3 = dist(fixed_point, q)

            pairs = [(d1, (fixed_point, p, q)),
                     (d2, (p, q, fixed_point)),
                     (d3, (fixed_point, q, p))]

            longest, (start, end, middle) = max(pairs, key=lambda x: x[0])
            # Ensure fixed_point is one of the endpoints, otherwise skip this triple
            if fixed_point != start and fixed_point != end:
                continue
            if start != fixed_point:
                start, end = end, start

            results.append((start, end,middle))

    return results


def build_quad_from_start_line(centers, first_line, area_tol=AREA_TOLERANCE,
                              side_ratio_thresh=1.5, angle_thresh=(30, 150),
                              used_points=None, lines=None, depth=0):
    if used_points is None:
        used_points = {first_line[0]['index'], first_line[1]['index'],first_line[2]['index']}
    else:
        used_points = used_points.copy()

    if lines is None:
        lines = [first_line]
    else:
        lines = lines.copy()

    # Perform early angle check if we have at least 2 sides (3 points)
    if depth >= 2:
        # Get the three points: prev, current, next
        p_prev = lines[-2][0]
        p_curr = lines[-2][1]  # same as lines[-1][0]
        p_next = lines[-1][1]

        v1 = np.array([p_prev['cx'], p_prev['cy']]) - np.array([p_curr['cx'], p_curr['cy']])
        v2 = np.array([p_next['cx'], p_next['cy']]) - np.array([p_curr['cx'], p_curr['cy']])
        angle_deg = angle_between(v1, v2)

        if not (angle_thresh[0] <= angle_deg <= angle_thresh[1]):
            return None  # early rejection

    if depth == 3:
        last_end = lines[-1][1]
        first_start = lines[0][0]

        if last_end != first_start:
            return None

        # Perform side ratio check
        side_lengths = [dist(line[0], line[1]) for line in lines]
        min_side = min(side_lengths)
        max_side = max(side_lengths)
        if max_side > side_ratio_thresh * min_side:
            return None

        # Reconstruct corner points
        quad_points = []
        for line in lines:
            if not quad_points or quad_points[-1]['index'] != line[0]['index']:
                quad_points.append(line[0])
        if lines[-1][1]['index'] != quad_points[0]['index']:
            quad_points.append(lines[-1][1])

        indices = [pt['index'] for pt in quad_points]
        if len(set(indices)) < 4:
            #print("Duplicate points found in quad:", indices)
            return None

        # Final angle check at the closing corner
        for i in range(4):
            prev = np.array([quad_points[i - 1]['cx'], quad_points[i - 1]['cy']])
            curr = np.array([quad_points[i]['cx'], quad_points[i]['cy']])
            next = np.array([quad_points[(i + 1) % 4]['cx'], quad_points[(i + 1) % 4]['cy']])
            angle_deg = angle_between(prev - curr, next - curr)

            if not (angle_thresh[0] <= angle_deg <= angle_thresh[1]):
                return None

        #print("Quad found!")
        return lines

    # Get fixed point to extend the side
    fixed_point = lines[-1][1]

    possible_next_lines = find_all_next_collinear_sides(
        centers,
        fixed_point,
        area_tol,
        used_points,
        allow_point=lines[0][0] if depth == 2 else None
    )

    for next_line in possible_next_lines:
        if next_line[1]['index'] in used_points and next_line[1] != lines[0][0]:
            continue

        used_points.add(next_line[1]['index'])
        new_lines = lines + [next_line]

        result = build_quad_from_start_line(
            centers,
            first_line,
            area_tol,
            side_ratio_thresh,
            angle_thresh,
            used_points=used_points,
            lines=new_lines,
            depth=depth + 1
        )

        if result is not None:
            return result

        used_points.remove(next_line[1]['index'])  # backtrack

    return None


def is_parallelogram(pts, length_ratio_tol=0.25, angle_tol_deg=10.0):
    """
    Check if a quadrilateral (given as 4 ordered points) is a parallelogram.
    
    Parameters:
        pts (list): 4 dicts, each with 'cx' and 'cy' keys.
        length_ratio_tol (float): Allowable ratio difference for side lengths (e.g., 0.25 = 25%)
        angle_tol_deg (float): Tolerance (in degrees) from 0° or 180° to consider sides parallel.
        
    Returns:
        True if it's a parallelogram, False otherwise.
    """

    if len(pts) != 4:
        raise ValueError("Exactly 4 points required")

    def vec(p1, p2):
        return np.array([p2['cx'] - p1['cx'], p2['cy'] - p1['cy']])

    def vec_len(v):
        return np.linalg.norm(v)

    def angle_between(v1, v2):
        cos_theta = np.clip(np.dot(v1, v2) / (vec_len(v1) * vec_len(v2) + 1e-10), -1.0, 1.0)
        return np.degrees(np.arccos(cos_theta))

    A, B, C, D = pts

    AB = vec(A, B)
    BC = vec(B, C)
    CD = vec(C, D)
    DA = vec(D, A)

    len_AB, len_CD = vec_len(AB), vec_len(CD)
    len_BC, len_DA = vec_len(BC), vec_len(DA)

    # Check side length ratios (more flexible than absolute pixel threshold)
    if not (1 - length_ratio_tol <= len_CD / len_AB <= 1 + length_ratio_tol):
        return False
    if not (1 - length_ratio_tol <= len_DA / len_BC <= 1 + length_ratio_tol):
        return False

    # Check angles between opposite sides (should be close to 0 or 180)
    angle_AB_CD = angle_between(AB, CD)
    angle_BC_DA = angle_between(BC, DA)

    if not (angle_AB_CD < angle_tol_deg or abs(angle_AB_CD - 180) < angle_tol_deg):
        return False
    if not (angle_BC_DA < angle_tol_deg or abs(angle_BC_DA - 180) < angle_tol_deg):
        return False

    return True

def is_middle_balanced(lines, ratio_thresh=1.3):
    """
    Ensures that for each line (start, end, middle), the distances from middle to start and middle to end
    are not overly imbalanced.

    Parameters:
        lines: List of 4 tuples (start, end, middle)
        ratio_thresh: Maximum allowed ratio between the longer and shorter distances

    Returns:
        True if all sides are balanced, False otherwise
    """
    def dist(p1, p2):
        return np.hypot(p2['cx'] - p1['cx'], p2['cy'] - p1['cy'])

    for i, (start, end, middle) in enumerate(lines):
        d1 = dist(start, middle)
        d2 = dist(end, middle)

        short = min(d1, d2)
        long = max(d1, d2)

        if short < 1e-3:
            #print(f"Line {i+1}: Too short to compare reliably.")
            return False  # avoid division by near-zero

        ratio = long / short
        if ratio > ratio_thresh:
            #print(f"Line {i+1} failed middle-balance check: ratio = {ratio:.2f} (limit {ratio_thresh})")
            return False

    return True


def draw_quad_on_frame(frame, p1, p2, p3, p4, color=(255, 0, 255), thickness=2):
    """
    Draws a quadrilateral connecting the 4 given center points on the frame.
    Points should be dicts with 'cx' and 'cy' keys.
    """
    try:
        pts = np.array([
            [p1['cx'], p1['cy']],
            [p2['cx'], p2['cy']],
            [p3['cx'], p3['cy']],
            [p4['cx'], p4['cy']],
        ], dtype=np.int32)
    except Exception as e:
        print("Failed to extract points for quad drawing:", e)
        return

    # Sort points clockwise (optional but consistent)
    center = np.mean(pts, axis=0)
    def angle(p): return np.arctan2(p[1] - center[1], p[0] - center[0])
    pts = sorted(pts, key=angle)

    pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))

    # Ensure color is a tuple of ints
    color = tuple(int(c) for c in color)

    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)

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
    dilated_edges = cv2.dilate(thresh, kernel, iterations=5)
    # Step 2: Close gaps inside sticker areas
    closed = cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    output = frame.copy()
    seen_contours = []
    centers = []
    index = 1

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000 or area > 10000:
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
                #print("Distance between 15 and 22 point",distance)
            if index == 21:
                p2 = np.array([cx, cy])
                distance = np.linalg.norm(p1 - p2)
                #print("Distance between 27 and 22 point",distance)
            if index == 4:
                p3 = np.array([cx, cy])

        # Draw center, text, and colored contour
        cv2.circle(output, (cx, cy), 3, (0, 0, 0), -1)
        cv2.putText(output, f"{index}", (cx - 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.drawContours(output, [approx], -1, bgr_color, 2)  # ← colored border
        index += 1
        
    center_points = [(c['cx'], c['cy']) for c in centers]



 
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

                #print(f"Triplet: {p1['index']}, {p2['index']}, {p3['index']}; Longest edge: {start['index']}->{end['index']}")


                result = build_quad_from_start_line(centers, first_line=(start, end, middle))

                if isinstance(result, str):
                    print("sorry")
                    #print(f"No quad chain for starting line from points {start['index']} to {end['index']}")
                elif result:
                    quad_lines = result  # result is a list of (start, end)
                    for i, line in enumerate(quad_lines):
                        start, end, middle = line
                        #print(f"  Line {i+1}:")
                        #print(f"    Start   idx {start['index']} -> ({start['cx']}, {start['cy']})")
                        #print(f"    End     idx {end['index']} -> ({end['cx']}, {end['cy']})")
                        #print(f"    Middle  idx {middle['index']} -> ({middle['cx']}, {middle['cy']})")
                    quad_points = [line[0] for line in quad_lines]
                    quad_points.append(quad_lines[-1][1])  # Close the loop

                    
                    #print("Final quad:")
                    #for pt in quad_points:
                        #print(f"  idx {pt['index']} -> ({pt['cx']}, {pt['cy']})")
                    #print(f"frame indx -> {frame_index}")
                    if is_parallelogram(quad_points[:4]) and is_middle_balanced(quad_lines):
                        #print("=> This parallelogram passed middle balance check!")
                        draw_quad_on_frame(output, *quad_points[:4])
                    #else:
                        #print("=> Parallelogram failed middle balance check.")


                                     

                
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
