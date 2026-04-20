import json
import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from roboflow import Roboflow
from dotenv import load_dotenv



BLUR_KSIZE = (9, 9)
LOWER_BLUE = np.array([85, 120, 140])
UPPER_BLUE = np.array([120, 230, 255])
DILATION_SIZE = 15
EROSION_SIZE = 15
CLOSE_KERNEL_SIZE = 15
EXPAND_AMOUNT = 500.0
ANGLE_THRESHOLD = 8.0
OUT_W, OUT_H = 2000, 1000


def get_image_metadata(target_filename:str, annotation_path:str = './data/_annotations.coco.json'):
    # loop but there only is a train split
    splits = ["train"]
    
    for split in splits:        
        # skip if the folder/file doesn't exist
        if not os.path.exists(annotation_path):
            continue
            
        with open(annotation_path, 'r') as f:
            coco_data = json.load(f)
        
        category_map = {cat['id']: cat['name'] for cat in coco_data['categories']}

        # search for the image in this split
        image_id = None
        for img in coco_data['images']:
            if img['file_name'].startswith(target_filename) or img.get('extra', {}).get('name') == target_filename:
                image_id = img['id']
                break
        
        # found in this split, process and return
        if image_id is not None:
            image_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id and category_map[ann['category_id']] in ['Striped', 'Solid', 'Black', 'Cue']]
            
            return {
                "filename": target_filename,
                "split_found_in": split,
                "total_balls": len(image_annotations),
                "ball_list": [category_map[ann['category_id']] for ann in image_annotations],
                "raw_bboxes": [ann['bbox'] for ann in image_annotations],
                "numbers": [ann['number'] for ann in image_annotations]
            }

    # if it loops through all splits and finds nothing (should not unless)
    return f"Metadata for {target_filename} not found in train, valid, or test sets."

def extract_table_contour(hsv_image, img_area):
    """
    Tries multiple HSV thresholds and validate the contour.
    """
    # threshold tiers
    # strict
    strict_lower = np.array([100, 150, 0])
    strict_upper = np.array([140, 255, 255])
    
    # loose
    loose_lower = np.array([95, 60, 20]) 
    loose_upper = np.array([145, 255, 255])
    
    thresholds_to_try = [
        ("Strict", strict_lower, strict_upper),
        ("Loose", loose_lower, loose_upper)
    ]
    
    # requirements to be considered a table
    MIN_AREA_RATIO = 0.125 # table must be at least 12.5% of the image
    MIN_EXTENT = 0.40     # table must fill at least 40% of its bounding box
    
    for name, lower, upper in thresholds_to_try:
        table_mask = cv2.inRange(hsv_image, lower, upper)
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            continue
            
        # get the largest blue contour should be the table
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # --- VALIDATION CHECKS ---
        # check size
        if area < (img_area * MIN_AREA_RATIO):
            print(f"[{name} Threshold] Failed: Largest contour too small ({area} vs {img_area * MIN_AREA_RATIO})")
            continue
            
        # check shape (Extent)
        x, y, w, h = cv2.boundingRect(largest_contour)
        rect_area = w * h
        extent = float(area) / rect_area
        
        if extent < MIN_EXTENT:
            print(f"[{name} Threshold] Failed: Extent too low ({extent:.2f}). Not rectangular enough.")
            continue
            
        print(f"[{name} Threshold] Success! Table found.")


        # clean mask where only the table will be
        surface_mask = np.zeros_like(table_mask)
        cv2.drawContours(surface_mask, [largest_contour], -1, 255, -1)

        return surface_mask, table_mask
        
    # If all loops fail
    return None, None


def detect_balls(img_path:str, show_plots:bool=False):
  
  img = cv2.imread(img_path)
  height, width = img.shape[:2]
  img = cv2.resize(img, (800, int(800 * height / width)))

  img_area = img.shape[0] * img.shape[1]

    
  #TABLE MASK
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
  surface_mask, table_mask = extract_table_contour(hsv_image=hsv, img_area=img_area)

  img_test = img.copy()
  gray = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)
    
  gray = cv2.bitwise_and(gray, gray, mask=table_mask)
  kernel = np.ones((15, 15), np.uint8)
  eroded_mask = cv2.erode(surface_mask, kernel, iterations=1)
  masked_frame = cv2.bitwise_and(gray, gray, mask=eroded_mask)
    
  # Blur first — reduces noise, improves circle detection significantly
  blurred = cv2.GaussianBlur(masked_frame, (3, 3), 0)
    
  # circles = cv2.HoughCircles(
  #     blurred,
  #     cv2.HOUGH_GRADIENT_ALT,
  #     dp=1.5,
  #     minDist=10,
  #     param1=100,
  #     param2=0.5,
  #     minRadius=4,
  #     maxRadius=20
  # )

  #---------OTHER APPROACH FOR FINDING THE BALLS---------
  # best performance
  circles = cv2.HoughCircles(
      blurred,
      cv2.HOUGH_GRADIENT,
      dp=1.2,
      minDist=15,
      param1=100,
      param2=15,
      minRadius=4,
      maxRadius=20
  )
  #------------------------------------------------------ 


  final_circles = []
  if circles is not None:
    median_test_radius = np.median(circles[0, :, 2])
    if show_plots:
      print(f"Median radius: {median_test_radius}")

      
    circles = np.round(circles[0]).astype(int)
    for x, y, r in circles:
      if show_plots:
        print(f"Circle found at ({x}, {y}) with radius {r}")
        
      # draw the outer circle
      cv2.circle(img_test, (x, y), r, (0, 255, 0), 2)
      # draw the center of the circle
      cv2.circle(img_test, (x, y), 2, (0, 0, 255), 3)
      final_circles.append([x, y, r])



  if show_plots:
  
    f, axarr = plt.subplots(1, 4, figsize=(20, 10))
    axarr[0].imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
    axarr[0].set_title("Blurred Image")
    axarr[0].axis("off")

    axarr[1].imshow(cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB))
    axarr[1].set_title("Detected Circles")
    axarr[1].axis("off")

    axarr[2].imshow(surface_mask, cmap="gray")
    axarr[2].set_title("Surface Mask")
    axarr[2].axis("off")

    axarr[3].imshow(eroded_mask, cmap="gray")
    axarr[3].set_title("Eroded Surface Mask")
    axarr[3].axis("off")

  return final_circles


def get_color_name(hue, sat, val):
    """Maps an OpenCV HSV value to a pool ball color."""
    # first handle Black and White (since they rely entirely on saturation and value not hue)
    if val < 60:
        return "Black"
    if sat < 100 and val > 120:
        return "White"
        
    # pool ball hues in OpenCV scale (0-179)
    if (hue >= 0 and hue < 5) or (hue > 165):
        return "Red"        # 3 or 11
    elif hue >= 5 and hue < 15:
        # orange and brown share similar hues. brown is just dark orange.
        if val < 150: return "Brown" # 7 or 15
        else: return "Orange"        # 5 or 13
    elif hue >= 16 and hue < 30:
        return "Yellow"     # 1 or 9
    elif hue >= 30 and hue < 100:
        return "Green"      # 6 or 14
    elif hue >= 100 and hue < 130: #descer a uns 115
        return "Blue"       # 2 or 10
    elif hue >= 130 and hue <= 165:
        return "Purple"     # 4 or 12
    
    return "Unknown"

def analyze_ball_color(img, cx, cy, r, show_plot=False):
    """
    Crops the ball, masks it, analyzes HSV colors, and plots a histogram.
    """
    height, width = img.shape[:2]
    
    # convert circle to bounding box (with safety checks so we don't go off-image)
    x_min = max(0, cx - r)
    y_min = max(0, cy - r)
    x_max = min(width, cx + r)
    y_max = min(height, cy + r)
    
    # crop the Bounding Box
    roi_bgr = img[y_min:y_max, x_min:x_max]
    
    # create a circular mask to ignore the table corners in the bounding box
    mask = np.zeros(roi_bgr.shape[:2], dtype=np.uint8)
    # the center of the circle in the cropped ROI is at (r, r) if it didn't hit an edge
    local_cx = cx - x_min
    local_cy = cy - y_min
    cv2.circle(mask, (local_cx, local_cy), r, 255, -1)
    
    # convert to HSV
    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    
    # Extract only the pixels inside our circular mask
    hsv_pixels = roi_hsv[mask == 255]
    
    if len(hsv_pixels) == 0:
        return "Unknown", "Unknown"
        
    hues = hsv_pixels[:, 0]
    sats = hsv_pixels[:, 1]
    vals = hsv_pixels[:, 2]
    
    # remove likely table-cloth pixels before classifying ball color
    table_bleed_mask = (hues >= 95) & (hues <= 125) & (sats > 60) & (vals > 100)
    valid_pixels_mask = np.zeros(mask.shape, dtype=np.uint8)
    valid_pixels_mask[mask == 255] = (~table_bleed_mask).astype(np.uint8) * 255
    hsv_pixels = hsv_pixels[~table_bleed_mask]

    if len(hsv_pixels) == 0:
        return "Unknown", "Unknown"
    
    hues = hsv_pixels[:, 0]
    sats = hsv_pixels[:, 1]
    vals = hsv_pixels[:, 2]
    
    # count White, Black, and Colored pixels
    # threshold might need adjustment
    white_mask = (sats < 100) & (vals > 120)
    black_mask = (vals < 60)
    
    white_count = np.sum(white_mask)
    black_count = np.sum(black_mask)
    total_pixels = len(hsv_pixels)
    
    # rest are 'colored' pixels
    color_mask = ~(white_mask | black_mask)
    color_hues = hues[color_mask]
    color_count = len(color_hues)
    
    # calculate the dominant Hue
    dominant_color = "Unknown"
    if color_count > 0:
        # calculate histogram of the colored pixels (0-180 for OpenCV Hue range)
        hist, bins = np.histogram(color_hues, bins=36, range=(0, 180))
        # find the bin with the most pixels
        peak_bin = np.argmax(hist)
        dominant_hue = (bins[peak_bin] + bins[peak_bin+1]) / 2
        if show_plot:
            print(f"Dominant Hue: {dominant_hue:.2f} (Bin {peak_bin}, Count {hist[peak_bin]}, peak bin range: {bins[peak_bin]:.2f}-{bins[peak_bin+1]:.2f})")
        # get a representative sat and val (medians of the colored pixels)
        med_sat = np.median(sats[color_mask])
        med_val = np.median(vals[color_mask])
        
        dominant_color = get_color_name(dominant_hue, med_sat, med_val)
    
    # solid vs striped vs cue vs black depends on the ration of white/black that exists in the ball
    white_ratio = white_count / total_pixels
    black_ratio = black_count / total_pixels
    
    if white_ratio > 0.70:
        ball_type = "Cue"
        final_color = "White"
    elif black_ratio > 0.30:
        ball_type = "Solid"
        final_color = "Black"
    # ff it has a significant amount of white and also a distinct color
    elif white_ratio > 0.15 and color_count > 0: 
        ball_type = "Striped"
        final_color = dominant_color
    # otherwise, it's mostly color
    else:
        ball_type = "Solid"
        final_color = dominant_color

    if show_plot:
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        
        # raw BGR crop
        axs[0].imshow(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB))
        axs[0].set_title(f"Crop: {ball_type} {final_color}")
        axs[0].axis('off')
        
        # mask isolated (the ball itself) after removing likely table-color pixels
        masked_bgr = cv2.bitwise_and(roi_bgr, roi_bgr, mask=valid_pixels_mask)
        axs[1].imshow(cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2RGB))
        axs[1].set_title("Masked Ball (Table Removed)")
        axs[1].axis('off')
        
        # Hue histogram (only for colored pixels)
        if color_count > 0:
            axs[2].hist(color_hues, bins=36, range=(0, 180), color='purple', alpha=0.7)
            axs[2].set_title("Hue Distribution (Only Color Pixels)")
            axs[2].set_xlabel("Hue (0-179)")
            axs[2].set_ylabel("Pixel Count")
        else:
            axs[2].text(0.5, 0.5, "No distinct color (Cue/Black)", ha='center')
            axs[2].axis('off')
            
        plt.tight_layout()
        plt.show()
        
    return ball_type, final_color

def detect_number (img_path, circles, show_plot=False):
    
    img = cv2.imread(img_path)

    height, width = img.shape[:2]
    img = cv2.resize(img, (800, int(800 * height / width)))

    final_number = []
    for cx, cy, r in circles:
        ball_type, color = analyze_ball_color(img, cx, cy, r, show_plot)

        if ball_type == 'Cue':
            final_number.append(0)
        elif ball_type == 'Solid' and color == 'Black':
            final_number.append(8)
        else:
            num_off_set = 0
            if ball_type == 'Striped':
                num_off_set = 8

            match color:
                case "Yellow":
                    final_number.append(1 + num_off_set)

                case "Blue":
                    final_number.append(2 + num_off_set)

                case "Red":
                    final_number.append(3 + num_off_set)

                case "Purple":
                    final_number.append(4 + num_off_set)

                case "Orange":
                    final_number.append(5 + num_off_set)

                case "Green":
                    final_number.append(6 + num_off_set)

                case "Brown":
                    final_number.append(7 + num_off_set)

                case _:
                    print("NO NUMBER DETECTED")
                    final_number.append(-1)
                
    return final_number


def calculate_iou(boxA, boxB):
    # boxes folow the format: [x_min, y_min, x_max, y_max]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute intersection area
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # compute areas of both bounding boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # calculate Intersection over Union
    if float(boxAArea + boxBArea - interArea) == 0:
        return 0.0
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def evaluate_image_detections(truth_boxes_coco, pred_circles, true_number_coco, pred_numbers, scale_x, scale_y, iou_threshold=0.5):
    """
    truth_boxes_coco: list of [x, y, width, height] from Roboflow
    pred_circles: list of [x_center, y_center, radius] from HoughCircles
    scale_x, scale_y: scaling factors to adjust ground truth to the resized 800px image
    """
    
    # convert COCO truth boxes to [x_min, y_min, x_max, y_max] apply scaling due to image resizing that happens in detect_balls()
    t_boxes = []
    for x, y, w, h in truth_boxes_coco:
        x_min = x * scale_x
        y_min = y * scale_y
        x_max = (x + w) * scale_x
        y_max = (y + h) * scale_y
        t_boxes.append([x_min, y_min, x_max, y_max])

    # convert predicted circles to [x_min, y_min, x_max, y_max]
    p_boxes = []
    for cx, cy, r in pred_circles:
        p_boxes.append([cx - r, cy - r, cx + r, cy + r])

    matched_ious = []
    detected_truth_indices = set()

    # will count the number of in/correctly predicted balls
    correct_number_count = 0
    incorrect_number_count = 0

    #has the preds and true numbers for when the predicted circle actualy matches a ball
    true_numbers_matched = []
    pred_numbers_matched = []
    matched_circles_id = []
    # match predictions to truth boxes
    for i, p_box in enumerate(p_boxes):
        best_iou = 0
        best_t_idx = -1
        
        for t_idx, t_box in enumerate(t_boxes):

            # if that ball was already claimed by another pred circle then it can't be claimed by this pred circle
            if t_idx in detected_truth_indices:
                continue

            iou = calculate_iou(p_box, t_box)
            if iou > best_iou:
                best_iou = iou
                best_t_idx = t_idx


        # just accept balls above the threshold
        if best_iou >= iou_threshold:
            matched_ious.append(best_iou)
            detected_truth_indices.add(best_t_idx)

            matched_circles_id.append(i)
            # best_t_idx is the index of the true bbox that was best matched to the curent pred bbox being analized
            if (true_number_coco[best_t_idx] == pred_numbers[i]):
                correct_number_count+=1
                true_numbers_matched.append(true_number_coco[best_t_idx])
                pred_numbers_matched.append(pred_numbers[i])
                
            else:
                # necessary because there might be balls predicted that are not balls and as such can't count for the incorrect classified
                incorrect_number_count+=1

                true_numbers_matched.append(true_number_coco[best_t_idx])
                pred_numbers_matched.append(pred_numbers[i])

        # print(f"Pred Box {i} Best IoU: {best_iou:.2f}") # ADD THIS


    true_count = len(truth_boxes_coco)
    pred_count = len(pred_circles)
    undetected_count = true_count - len(detected_truth_indices)

    return {
        "true_count": true_count,
        "pred_count": pred_count,
        "correct_number_count": correct_number_count,
        "incorrect_number_count": incorrect_number_count,
        "true_numbers_matched": true_numbers_matched,
        "pred_numbers_matched": pred_numbers_matched,
        "undetected_count": undetected_count,
        "matched_ious": matched_ious,
        "matched_circles": matched_circles_id
    }



def find_images(folder):
    paths = []
    if not os.path.isdir(folder):
        return []
    for filename in os.listdir(folder):
        paths.append(os.path.join(folder, filename))
    return sorted(paths)

def blur_image(image):
    return cv2.GaussianBlur(image, ksize=BLUR_KSIZE, sigmaX=0)

def build_blue_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)

def refine_mask(mask):
    d_kernel = np.ones((DILATION_SIZE, DILATION_SIZE), np.uint8)
    e_kernel = np.ones((EROSION_SIZE, EROSION_SIZE), np.uint8)
    close_kernel = np.ones((CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE), np.uint8)

    mask = cv2.dilate(mask, d_kernel, iterations=1)
    mask = cv2.dilate(mask, e_kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    mask = cv2.dilate(mask, d_kernel, iterations=1)
    return mask


def find_largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)

def edge_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = np.degrees(np.arctan2(dy, dx))
    return (angle + 180) % 180

def angle_diff(a, b):
    d = abs(a - b)
    return min(d, 180 - d)


def merge_collinear_segments(pts, angle_threshold=ANGLE_THRESHOLD):
    pts = np.asarray(pts, dtype=float)
    n = len(pts)
    if n < 2:
        return []

    edges = []
    for i in range(n):
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        angle = edge_angle(p1, p2)
        length = np.linalg.norm(p2 - p1)
        edges.append({'start': p1, 'end': p2, 'angle': angle, 'len': length})

    groups = []
    current = {
        'start': edges[0]['start'],
        'end': edges[0]['end'],
        'angle': edges[0]['angle'],
        'len': edges[0]['len'],
        'angle_weighted': edges[0]['angle'] * edges[0]['len'],
    }

    for edge in edges[1:]:
        if angle_diff(edge['angle'], current['angle']) <= angle_threshold:
            current['end'] = edge['end']
            current['len'] += edge['len']
            current['angle_weighted'] += edge['angle'] * edge['len']
            current['angle'] = current['angle_weighted'] / current['len']
        else:
            groups.append(current)
            current = {
                'start': edge['start'],
                'end': edge['end'],
                'angle': edge['angle'],
                'len': edge['len'],
                'angle_weighted': edge['angle'] * edge['len'],
            }

    groups.append(current)

    if len(groups) > 1 and angle_diff(groups[0]['angle'], groups[-1]['angle']) <= angle_threshold:
        groups[0]['start'] = groups[-1]['start']
        groups[0]['len'] += groups[-1]['len']
        groups[0]['angle_weighted'] += groups[-1]['angle'] * groups[-1]['len']
        groups[0]['angle'] = groups[0]['angle_weighted'] / groups[0]['len']
        groups.pop()

    for g in groups:
        g.pop('angle_weighted', None)

    return groups


def line_from_segment(seg):
    p1, p2 = seg['start'], seg['end']
    x1, y1 = p1
    x2, y2 = p2
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    return a, b, c

def intersect_line_segments(seg1, seg2):
    a1, b1, c1 = line_from_segment(seg1)
    a2, b2, c2 = line_from_segment(seg2)
    det = a1 * b2 - a2 * b1
    if np.isclose(det, 0.0):
        return None
    x = (b1 * c2 - b2 * c1) / det
    y = (c1 * a2 - c2 * a1) / det
    return np.array([x, y], dtype="float32")

def expand_segment(seg, amount=EXPAND_AMOUNT):
    p1, p2 = seg['start'], seg['end']
    direction = p2 - p1
    length = np.linalg.norm(direction)
    if length == 0:
        return {'start': p1.copy(), 'end': p2.copy(), 'len': seg['len']}
    unit = direction / length
    return {
        'start': p1 - unit * amount,
        'end': p2 + unit * amount,
        'len': seg['len'],
    }

def midpoint(line):
    return (line['start'] + line['end']) / 2


def reorder_corners_top_right(corners):
    corners = corners.copy()
    sort_idx = np.lexsort((corners[:, 1], -corners[:, 0]))
    top_right_idx = sort_idx[0]
    return np.roll(corners, -top_right_idx, axis=0)

def get_table_corners(hull):
    pts = hull.reshape(-1, 2).astype(float)
    merged_lines = merge_collinear_segments(pts)
    if len(merged_lines) < 4:
        raise ValueError(f"Only found {len(merged_lines)} merged hull segments. Check hull quality.")

    top_4 = sorted(merged_lines, key=lambda x: x['len'], reverse=True)[:4]
    expanded_lines = [expand_segment(line) for line in top_4]
    center = np.mean([midpoint(line) for line in expanded_lines], axis=0)
    ordered_lines = sorted(
        expanded_lines,
        key=lambda line: np.arctan2(midpoint(line)[1] - center[1], midpoint(line)[0] - center[0])
    )

    raw_corners = []
    for i in range(4):
        seg1 = ordered_lines[i]
        seg2 = ordered_lines[(i + 1) % 4]
        pt = intersect_line_segments(seg1, seg2)
        if pt is None:
            pt = (midpoint(seg1) + midpoint(seg2)) / 2
        raw_corners.append(pt.astype("float32"))

    return np.stack(raw_corners, axis=0)

def warp_table(image, corners):
    dst_pts = np.array(
        [[OUT_W - 1, 0], [OUT_W - 1, OUT_H - 1], [0, OUT_H - 1], [0, 0]],
        dtype="float32",
    )
    corners = reorder_corners_top_right(corners)
    matrix = cv2.getPerspectiveTransform(corners, dst_pts)
    return cv2.warpPerspective(image, matrix, (OUT_W, OUT_H))

def process_image(image_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    blurred = blur_image(image)
    mask = build_blue_mask(blurred)
    mask = refine_mask(mask)

    contour = find_largest_contour(mask)
    if contour is None:
        raise RuntimeError(f"No contour found for image: {image_path}")

    hull = cv2.convexHull(contour, clockwise=True)
    corners = get_table_corners(hull)
    warped = warp_table(image, corners)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, warped)
    return output_path


def warp_pipeline():
    data_folder = os.path.join(os.path.dirname(__file__), "data", "development_set")
    output_folder = os.path.join(os.path.dirname(__file__), "flattened")

    image_paths = find_images(data_folder)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {data_folder}")

    print(f"Found {len(image_paths)} images. Processing...")
    for image_path in image_paths:
        name = os.path.basename(image_path)
        output_path = os.path.join(output_folder, name)
        try:
            saved_path = process_image(image_path, output_path)
            print(f"Saved warped image: {saved_path}")
        except Exception as exc:
            print(f"Failed to process {name}: {exc}")





if __name__ == "__main__":

    TEST_DATA_PATH = "./data/example_json/input.json"

    with open(TEST_DATA_PATH, 'r') as f:
        paths = json.load(f)
    ims_path = paths['image_path']

    total_images = 0
    total_absolute_error = 0
    exact_count_matches = 0
    total_truth_balls = 0
    total_undetected_balls = 0
    total_correct_number = 0
    total_incorrect_number = 0
    all_ious = []
    y = []
    y_pred = []

    output_json_data = []

    for path in ims_path:

        im_name = re.findall(pattern="[0-9]+._png", string=path)[0]
        img_full_path = os.path.join('./data', path)

        # get metadata/ground truth
        metadata = get_image_metadata(target_filename=im_name)
        if isinstance(metadata, str): 
            print(metadata) # Metadata not found
            continue

        # original image dimensions to calculate scaling factors
        original_img = cv2.imread(img_full_path)

        #error reading image
        if original_img is None:
            continue
        orig_height, orig_width = original_img.shape[:2]

        # dimensions detect_balls function resizes to:
        resized_width = 800
        resized_height = int(800 * orig_height / orig_width)

        scale_x = resized_width / orig_width
        scale_y = resized_height / orig_height

        #-----------------DETECTION CALL-----------------
        circles = detect_balls(img_path=img_full_path, show_plots=False)
        numbers = detect_number(img_path=img_full_path, circles=circles, show_plot=False)
        #-----------------DETECTION CALL-----------------





        #-----------------Evaluation CALL-----------------
        stats = evaluate_image_detections(
            truth_boxes_coco=metadata["raw_bboxes"],
            pred_circles=circles,
            true_number_coco= metadata["numbers"],
            pred_numbers = numbers,
            scale_x=scale_x, 
            scale_y=scale_y,
            iou_threshold=0.2 # could be tunned ask professor
        )
        #-----------------Evaluation CALL-----------------

        total_images += 1
        total_absolute_error += abs(stats["pred_count"] - stats["true_count"])

        if stats["pred_count"] == stats["true_count"]:
            exact_count_matches += 1

        total_truth_balls += stats["true_count"]
        total_undetected_balls += stats["undetected_count"]
        total_correct_number += stats["correct_number_count"]
        total_incorrect_number += stats["incorrect_number_count"]
        all_ious.extend(stats["matched_ious"])

        if "true_numbers_matched" in stats:
            y.extend(stats["true_numbers_matched"])
        if "pred_numbers_matched" in stats:
            y_pred.extend(stats["pred_numbers_matched"])

        print(f"Processed {im_name} | True: {stats['true_count']} | Pred: {stats['pred_count']} | Undetected: {stats['undetected_count']}")



        image_balls_data = []

        # convert circles to normalized bounding boxes
        if circles is not None:
            for i, (cx, cy, r) in enumerate(circles):


                # calculate normalized coordinates (0.0 to 1.0)
                # ensure boxes don't go outside image boundaries
                xmin = max(0.0, (cx - r) / resized_width)
                xmax = min(1.0, (cx + r) / resized_width)
                ymin = max(0.0, (cy - r) / resized_height)
                ymax = min(1.0, (cy + r) / resized_height)

                # Match circle index to number prediction
                if i in stats["matched_circles"]:
                    ball_number = numbers[i]
                else:
                    ball_number = -1

                image_balls_data.append({
                    "number": int(ball_number),
                    "xmin": float(xmin),
                    "xmax": float(xmax),
                    "ymin": float(ymin),
                    "ymax": float(ymax)
                })

        output_json_data.append({
            "image_path": path,
            "num_balls": len(image_balls_data),
            "balls": image_balls_data
        })


    #-----------------METRICS CALCULATION-----------------
    print("\n" + "="*30)
    print("FINAL PIPELINE METRICS")
    print("="*30)

    # Ball Count MAE
    mae = total_absolute_error / total_images if total_images > 0 else 0
    print(f"Ball Count MAE: {mae:.2f} balls")

    # Ball Count Accuracy
    accuracy = (exact_count_matches / total_images) * 100 if total_images > 0 else 0
    print(f"Ball Count Accuracy (Exact Match): {accuracy:.2f}%")

    # BBox Detection IOU
    mean_iou = np.mean(all_ious) if all_ious else 0
    print(f"Average BBox IOU (True Positives): {mean_iou:.4f}")

    # Average % of undetected balls
    undetected_pct = (total_undetected_balls / total_truth_balls) * 100 if total_truth_balls > 0 else 0
    print(f"Percentage of Undetected Balls: {undetected_pct:.2f}%")

    # Average % of correct numbered balls
    correct_number_pct = (total_correct_number / (total_correct_number+total_incorrect_number)) * 100 if (total_correct_number+total_incorrect_number) > 0 else 0
    print(f"Percentage of correct numbers indentified in Balls: {correct_number_pct:.2f}%")

    # Average % of incorrect numbered balls
    incorrect_number_pct = (total_incorrect_number / (total_correct_number+total_incorrect_number)) * 100 if (total_correct_number+total_incorrect_number) > 0 else 0
    print(f"Percentage of incorrect numbers indentified in Balls: {incorrect_number_pct:.2f}%")


    # --- NEW: Save the JSON file ---
    OUTPUT_JSON_PATH = "predictions.json"
    with open(OUTPUT_JSON_PATH, "w") as outfile:
        json.dump(output_json_data, outfile, indent=4)

    print(f"\nSaved predictions to {OUTPUT_JSON_PATH}")
    print("="*30)


    print("------STARING WARPING PIPELINE------")
    warp_pipeline()