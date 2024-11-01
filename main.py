import cv2
import argparse

from ultralytics import YOLO
import supervision as sv
import numpy as np

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution", 
        default=[1280, 720], 
        nargs=2, 
        type=int
    )
    args = parser.parse_args()
    return args

FRAME_ZONE = np.array([
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1]
])


def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("best.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    zone_polygon = (FRAME_ZONE * np.array(args.webcam_resolution)).astype(int)
    zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=tuple(args.webcam_resolution))
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone, 
        color=sv.Color.red(),
        thickness=2,
        text_thickness=4,
        text_scale=2
    )

    while True:
        ret, frame = capture.read()

        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)
        
        low_confidence_detected = False  # Flag for low confidence detection
        labels = []

        for _, confidence, class_id, _ in detections:
            if confidence < 0.5:
                low_confidence_detected = True
            labels.append(f"{model.model.names[class_id]} {confidence:0.2f}")

        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections, 
            labels=labels
        )

        zone.trigger(detections=detections)
        frame = zone_annotator.annotate(scene=frame)
        
        object_count = len(detections)
        cv2.putText(frame, f'Objects: {object_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Draw notification box at the bottom left
        if low_confidence_detected:
            notif_height, notif_width = 100, 300
            x_start, y_start = 10, frame_height - notif_height - 10
            cv2.rectangle(frame, (x_start, y_start), (x_start + notif_width, y_start + notif_height), (0, 0, 0), -1)
            cv2.putText(frame, "Warning!! Low confidence", 
                        (x_start + 10, y_start + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "detections! Adjust angle.", 
                        (x_start + 10, y_start + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow("YOLOv8 Live Detection", frame)

        if (cv2.waitKey(30) == 27):
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
