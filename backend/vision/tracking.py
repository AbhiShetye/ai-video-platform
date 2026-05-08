def track_objects(detection_results):
    tracked = {}

    for frame_data in detection_results:
        frame_id = frame_data["frame_id"]

        for i, obj in enumerate(frame_data["objects"]):
            object_id = f"{obj['class']}_{i}"

            if object_id not in tracked:
                tracked[object_id] = {
                    "object_id": object_id,
                    "label": obj["class"],
                    "frames": []
                }

            tracked[object_id]["frames"].append({
                "frame_id": frame_id,
                "bbox": obj["bbox"]
            })

    return list(tracked.values())