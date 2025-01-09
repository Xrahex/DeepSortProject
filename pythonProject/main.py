import cv2
import numpy as np
import csv
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment


class Main:
    def __init__(self, video_path, model_id="yolov8s", output_file="tracked_objects.csv"):
        self.video_path = video_path
        self.video_player = VideoPlayer(video_path)
        self.model = YOLO(model_id)
        self.tracker = Tracker(max_inactive_frames=10)
        self.output_file = output_file
        self.tracked_objects = []

    def process_video(self):
        while True:
            frame = self.video_player.get_next_frame()
            if frame is None:
                break

            results = self.model(frame)

            detections = []
            for result in results:
                xyxy = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()
                for i in range(len(xyxy)):
                    if class_ids[i] == 0:
                        detections.append([xyxy[i], confidences[i], int(class_ids[i])])

            detections = group_and_merge_detections(detections, iou_threshold=0.5)

            tracked_objects = self.tracker.track_objects(detections)

            annotated_image = self.annotate_frame(frame, tracked_objects)
            self.tracked_objects.extend(tracked_objects)

            cv2.imshow('YOLOv8 - Video Processing', annotated_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.save_tracked_objects()

    def annotate_frame(self, frame, tracked_objects):
        for obj in tracked_objects:
            x1, y1, x2, y2, confidence, class_id, object_id = obj
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"ID: {object_id} Confidence: {confidence:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame

    def save_tracked_objects(self):
        with open(self.output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['ID', 'Frame', 'x1', 'y1', 'x2', 'y2', 'Confidence'])
            for obj in self.tracked_objects:
                writer.writerow(obj)


class VideoPlayer:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)

    def get_next_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame


class Tracker:
    def __init__(self, max_distance=50, max_inactive_frames=10):
        self.max_distance = max_distance
        self.max_inactive_frames = max_inactive_frames
        self.objects = {}
        self.next_object_id = 1

    def track_objects(self, detections):
        if not self.objects:
            for det in detections:
                self.objects[self.next_object_id] = [det, 0]
                self.next_object_id += 1
        else:
            object_ids = list(self.objects.keys())
            object_locations = np.array([obj[0][0] for obj in self.objects.values()])
            detection_locations = np.array([det[0] for det in detections])

            cost_matrix = np.zeros((len(object_locations), len(detection_locations)))
            for i, obj_loc in enumerate(object_locations):
                for j, det_loc in enumerate(detection_locations):
                    cost_matrix[i, j] = np.linalg.norm(obj_loc - det_loc)

            row_indices, col_indices = linear_sum_assignment(cost_matrix)

            matched_indices = set()
            for row, col in zip(row_indices, col_indices):
                if cost_matrix[row, col] < self.max_distance:
                    obj_id = object_ids[row]
                    self.objects[obj_id] = [detections[col], 0]
                    matched_indices.add(col)

            unmatched_detections = [i for i in range(len(detections)) if i not in matched_indices]
            for idx in unmatched_detections:
                self.objects[self.next_object_id] = [detections[idx], 0]
                self.next_object_id += 1

            inactive_object_ids = []
            for obj_id in list(self.objects.keys()):
                self.objects[obj_id][1] += 1
                if self.objects[obj_id][1] > self.max_inactive_frames:
                    inactive_object_ids.append(obj_id)

            for obj_id in inactive_object_ids:
                del self.objects[obj_id]

        return [[*det[0], det[1], det[2], obj_id] for obj_id, (det, _) in self.objects.items()]


def group_and_merge_detections(detections, iou_threshold=0.5):
    def compute_iou(box1, box2):
        x1, y1, x2, y2 = box1
        x1_, y1_, x2_, y2_ = box2

        inter_x1 = max(x1, x1_)
        inter_y1 = max(y1, y1_)
        inter_x2 = min(x2, x2_)
        inter_y2 = min(y2, y2_)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_ - x1_) * (y2_ - y1_)

        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    detections = [[tuple(det[0]), det[1], det[2]] for det in detections]

    merged_detections = []
    while detections:
        base_det = detections.pop(0)
        base_box, base_conf, base_class = base_det

        group = [base_box]
        to_remove = []
        for det in detections:
            other_box, other_conf, other_class = det
            if compute_iou(base_box, other_box) > iou_threshold:
                group.append(other_box)
                to_remove.append(det)

        detections = [det for det in detections if det not in to_remove]

        x1 = min([box[0] for box in group])
        y1 = min([box[1] for box in group])
        x2 = max([box[2] for box in group])
        y2 = max([box[3] for box in group])

        confidence = max([base_conf] + [det[1] for det in to_remove])
        merged_detections.append([[x1, y1, x2, y2], confidence, base_class])

    return merged_detections


video_path = "video.mp4"
output_file = "tracked_objects.csv"

main = Main(video_path, model_id="yolov8s", output_file=output_file)
main.process_video()

cv2.destroyAllWindows()
