import os
import cv2
import numpy as np

BLUR_KSIZE = (9, 9)
LOWER_BLUE = np.array([85, 120, 140])
UPPER_BLUE = np.array([120, 230, 255])
DILATION_SIZE = 15
EROSION_SIZE = 15
CLOSE_KERNEL_SIZE = 15
EXPAND_AMOUNT = 500.0
ANGLE_THRESHOLD = 8.0
OUT_W, OUT_H = 1000, 500

def find_images(folder):
    extensions = (".jpg", ".jpeg", ".png")
    paths = []
    if not os.path.isdir(folder):
        return []
    for filename in os.listdir(folder):
        if filename.lower().endswith(extensions):
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


def main():
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

if __name__ == '__main__':
    main()
    