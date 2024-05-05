import os
import cv2
import numpy as np


class Painter():
    def __init__(self,
                 labels: list,
                 labels_to_paint: list,
                 bbox_thickness: int,
                 ):
        rng = np.random.default_rng(3)
        self.colors = rng.uniform(0, 255, size=(len(labels), 3))
        self.labels = labels
        self.labels_to_paint = labels_to_paint
        self.bbox_thickness = bbox_thickness


    def __call__(self, frame: np.ndarray, 
                    boxes: list,
                    scores: list = [],
                    class_ids: list = [],
                    track_ids: list = [],) -> np.ndarray:
        
        frame = self.draw_boxes(frame,
                                boxes,
                                scores,
                                class_ids,
                                track_ids)
        return frame

    def draw_boxes(self, 
                    frame: np.ndarray, 
                    boxes: list,
                    scores: list = [],
                    class_ids: list = [],
                    track_ids: list = [],) -> np.ndarray:

        img_height, img_width = frame.shape[:2]
        font_size = min([img_height, img_width]) * 0.0006
        text_thickness = int(min([img_height, img_width]) * 0.001)

        scores = scores if len(scores) else [-1] * len(boxes)
        class_ids = class_ids if len(class_ids) else [-1] * len(boxes)
        track_ids = track_ids if len(track_ids) else [-1] * len(boxes)

        for box, score, class_id, track_id in zip(boxes, scores, class_ids, track_ids):
            color = self.colors[class_id]

            x1, y1, x2, y2 = [int(var) for var in box]

            caption = ''
            if class_id >= 0:
                caption = self.labels[class_id]
                if not (self.labels[class_id] in self.labels_to_paint):
                    continue
            if score >= 0:
                caption = f"{caption} {int(score * 100)}%"
            if track_id >= 0:
                caption = f"{caption} track: {track_id}"

            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.bbox_thickness)
            frame = self.draw_text(frame, caption, box, color, font_size, text_thickness)

        return frame
    
    def draw_text(self, frame: np.ndarray,
                   text: str, 
                   box: list, 
                   color: tuple = (0, 0, 255),
                   font_size: float = 0.001, 
                   text_thickness: int = 2) -> np.ndarray:
        
        x1, y1, x2, y2 = [int(var) for var in box]

        (tw, th), _ = cv2.getTextSize(text=text, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=font_size, thickness=text_thickness)
        th = int(th * 1.2)

        cv2.rectangle(frame, (x1, y1),
                    (x1 + tw, y1 - th), color, -1)

        return cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), text_thickness, cv2.LINE_AA)


