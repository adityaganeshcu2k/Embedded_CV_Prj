import cv2
import numpy as np

cnt_green = 0;
# Load image
image = cv2.imread("rubiks.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Apply GaussianBlur to reduce noise
#blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Option 1: Adaptive threshold to boost grid lines
#adaptive = cv2.adaptiveThreshold(
#    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
#)


# Apply Canny on the processed image
edges = cv2.Canny(gray, 5, 100)

kernel = np.ones((2, 2), np.uint8)  # You can also try (5,5)
dilated_edges = cv2.dilate(edges, kernel, iterations=1)


#dilated_edges = cv2.dilate(edges, kernel, iterations=1)


# Find contours
contours, _ = cv2.findContours(dilated_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

output = image.copy()
hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)

seen_contours = []

index = 1  # Start indexing contours
# Step 3: Loop through all contours and filter by HSV
for cnt in contours:

    area = cv2.contourArea(cnt)

    if area < 400 or area > 6000:  # skip small noise
        continue
    # Create a mask for the current contour
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, thickness=-1)  # Fill the contour in the mask
    # Calculate average HSV value in ROI
    avg_hsv = cv2.mean(hsv, mask=mask)[:3]
    h_val, s_val, v_val = avg_hsv

    M = cv2.moments(cnt)
    if M["m00"] == 0:
        continue

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # --- Check if similar (area and center) contour already seen ---
    is_duplicate = False
    for seen_area, seen_cx, seen_cy in seen_contours:
        if abs(cx - seen_cx) < 10 and abs(cy - seen_cy) < 10:
            is_duplicate = True
            break
    if is_duplicate:
        continue

    # Add this contour to seen list
    seen_contours.append((area, cx, cy))
    

    # Draw center
    cv2.circle(output, (cx, cy), 3, (0, 0, 0), -1)

        # Draw the contour number
    cv2.putText(output, str(index), (cx - 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    index += 1
    # Create mask for HSV
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, thickness=-1)
    avg_hsv = cv2.mean(hsv, mask=mask)[:3]
    h_val, s_val, v_val = avg_hsv

    print(f"HSV: H={h_val:.1f}, S={s_val:.1f}, V={v_val:.1f}")
    print("CX :",cx)
    print("CY :",cy)
    print("Index :",index-1)
    # Color detection and drawing
    if 100 <= h_val <= 140 and s_val >= 100:
        print("Blue detected")
        cv2.drawContours(output, [cnt], -1, (255, 0, 0), 2)  # Blue
    elif 10 < h_val <= 25 and s_val >= 100:
        print("Orange detected")
        cv2.drawContours(output, [cnt], -1, (0, 165, 255), 2)  # Orange (BGR)
    elif 26 <= h_val <= 45 and s_val >= 100:
        print("Yellow detected")
        cv2.drawContours(output, [cnt], -1, (0, 255, 255), 2)
    elif 35 <= h_val <= 85 and s_val >= 100:
        print("Green detected")
        cnt_green += 1
        cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)  # Green
    elif 0 <= h_val <= 10 or h_val >= 160:
        print("Red detected")
        cv2.drawContours(output, [cnt], -1, (0, 0, 255), 2)  # Red
        
print("Num of green :",cnt_green)
# Show the images
cv2.imshow("Gray", gray)
#cv2.imshow("Adaptive Threshold", adaptive)
cv2.imshow("Canny Edges", dilated_edges)
cv2.imshow("Contours", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
