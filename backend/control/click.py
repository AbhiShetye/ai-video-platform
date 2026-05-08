def select_object(click_x, click_y, detections):
    selected = None
    min_distance = float('inf')

    for detection in detections:
        object_id = detection["object_id"]
        bbox = detection["bbox"]  # [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox

        # Check if click is inside bbox
        if x1 <= click_x <= x2 and y1 <= click_y <= y2:
            return {
                "object_id": object_id,
                "label": detection.get("label") or detection.get("class", ""),
                "bbox": bbox
            }

        # If not inside, find nearest bbox center
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        distance = ((click_x - center_x) ** 2 + (click_y - center_y) ** 2) ** 0.5

        if distance < min_distance:
            min_distance = distance
            selected = {
                "object_id": object_id,
                "label": detection.get("label") or detection.get("class", ""),
                "bbox": bbox
            }

    return selected