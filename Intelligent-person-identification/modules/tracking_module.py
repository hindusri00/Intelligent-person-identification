import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from modules.attribute_module import AttributeAnalyzer
from modules.reid_module import PersonReIdentifier
import time
class RealTimeTracker:
    def __init__(self, model_path="yolo11n.pt"):
        self.model = YOLO(model_path)
        self.attribute_analyzer = AttributeAnalyzer()
        self.reid = PersonReIdentifier()
        self.frame_count = 0
        self.reid_cache = {}
        self.attribute_cache = {}
        self.target_embedding = None
        self.target_similarity_threshold = 0.80
        self.target_cache = {}
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            nms_max_overlap=1.0,
            max_cosine_distance=0.2,
            embedder="mobilenet",
            half=True
        )
    def set_target(self, embedding):
        self.target_embedding = embedding
        self.target_cache = {}
        print("[Target] Target person set successfully")

    def process_frame(self, frame):
        start_time = time.time()
        self.frame_count += 1

        results = self.model(
            frame,
            verbose=False,
            device=0,
            half=True,
            imgsz=640
        )
        detections = []
        carried_accessories = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls_id == 0 and conf > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    detections.append(([x1, y1, w, h], conf, 'person'))
                elif cls_id in [24, 26] and conf > 0.4:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    carried_accessories.append((x1, y1, x2, y2, self.model.names[cls_id]))

        tracks = self.tracker.update_tracks(detections, frame=frame)
        active_targets = []

        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            person_crop = frame[y1:y2, x1:x2]
            bbox_height = y2 - y1
            # Run Re-ID only every 5 frames for each track
            # Target-person matching
            if self.target_embedding is not None:

                if track_id not in self.target_cache or self.frame_count % 15 == 0:
                    embedding = self.reid.extract_embedding(person_crop)

                    similarity = self.reid.calculate_similarity(
                        embedding,
                        self.target_embedding
                    )

                    self.target_cache[track_id] = {
                        "is_target": similarity >= self.target_similarity_threshold,
                        "similarity": similarity
                    }

                target_result = self.target_cache[track_id]

                # Ignore everyone who does not match the target
                if not target_result["is_target"]:
                    continue

                person_id = "TARGET"
                reid_similarity = target_result["similarity"]

            else:

                # Normal multi-person Re-ID mode
                if track_id not in self.reid_cache or self.frame_count % 5 == 0:
                    reid_result = self.reid.identify(person_crop)

                    self.reid_cache[track_id] = {
                        "person_id": reid_result["person_id"],
                        "similarity": reid_result["similarity"]
                    }

                reid_result = self.reid_cache[track_id]

                person_id = reid_result["person_id"]
                reid_similarity = reid_result["similarity"]

            # Calculate attributes only every 10 frames
            if track_id not in self.attribute_cache or self.frame_count % 10 == 0:
                self.attribute_cache[track_id] = (
                    self.attribute_analyzer.extract_all_attributes(
                        person_crop,
                        bbox_height
                    )
                )

            attr = self.attribute_cache[track_id]

            has_bag = "None"
            for ax1, ay1, ax2, ay2, item_type in carried_accessories:
                if not (x2 < ax1 or x1 > ax2 or y2 < ay1 or y1 > ay2):
                    has_bag = item_type
                    break

            # Build Dynamic Tag List (Shows Active Detections Only)
            detected_tags = [
                f"Track #{track_id}",
                f"Person ID: {person_id}"
            ]

            if self.target_embedding is not None:
                detected_tags.append(
                    f"Match: {reid_similarity:.2f}"
                )

            if attr["estimated_height"] != "Unknown":
                detected_tags.append(f"Ht: {attr['estimated_height']}")
            if attr["hair_color"] != "Unknown":
                detected_tags.append(f"Hair: {attr['hair_color']}")
            if attr["shirt_color"] != "Unknown":
                detected_tags.append(f"Shirt: {attr['shirt_color']}")
            if attr["pant_color"] != "Unknown":
                detected_tags.append(f"Pant: {attr['pant_color']}")
            if attr["shoe_color"] != "Unknown":
                detected_tags.append(f"Shoes: {attr['shoe_color']}")
            
            if attr["mask"] == "Yes":
                detected_tags.append(f"Mask: {attr['mask_color']}")
            if attr["spectacles"] == "Yes":
                detected_tags.append("Glasses")
            if attr["wristband"] == "Yes":
                detected_tags.append("Wristband")
            if attr["tattoo"] == "Yes":
                detected_tags.append("Tattoo")
            if has_bag != "None":
                detected_tags.append(f"Bag: {has_bag}")

            active_targets.append({
                "track_id": track_id,
                "person_id": person_id,
                "reid_similarity": reid_similarity,
                "bbox": [x1, y1, x2, y2],
                **attr,
                "accessory": has_bag
            })

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = " | ".join(detected_tags)
            cv2.putText(frame, label_text, (x1, max(20, y1 - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
        total_time = time.time() - start_time
        print(f"[PERF] Frame: {total_time:.3f}s")
        return frame, active_targets