import cv2
import numpy as np
import onnxruntime

from yolo_utils import xywh2xyxy, multiclass_nms


class YOLOv8:
    def __init__(self, 
                 weights_path: str, 
                 id2thr: dict = {},
                 id2min_wh: dict = {},
                 id2max_wh: dict = {},
                 conf_threshold: float = 0.7, 
                 iou_threshold: float = 0.5,):
        
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.id2thr = id2thr
        self.id2min_wh = id2min_wh
        self.id2max_wh = id2max_wh

        self.initialize_model(weights_path)

    def __call__(self, image: np.ndarray):
        return self.detect_objects(image)
    

    def initialize_model(self, path: str):
        self.session = onnxruntime.InferenceSession(path,
                            providers=onnxruntime.get_available_providers())
        self.get_input_details()
        self.get_output_details()


    def detect_objects(self, image: np.ndarray):
        input_tensor = self.prepare_input(image)

        outputs = self.inference(input_tensor)

        self.boxes, self.scores, self.class_ids = self.process_output(outputs)

        return self.boxes, self.scores, self.class_ids

    def prepare_input(self, image: np.ndarray) -> np.ndarray:
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor


    def inference(self, input_tensor: np.ndarray) -> np.ndarray:
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        return np.array(outputs)

    def process_output(self, output: np.ndarray):
        predictions = np.squeeze(output[0]).T

        scores = np.max(predictions[:, 4:], axis=1)

        if len(self.id2thr) == 0:
            predictions = predictions[scores > self.conf_threshold, :]
            scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], []

        class_ids = np.argmax(predictions[:, 4:], axis=1)

        boxes = self.extract_boxes(predictions)
        indices = multiclass_nms(boxes, scores, class_ids, self.iou_threshold, self.id2thr)

        boxes, scores, class_ids = self.filter_size(boxes[indices], scores[indices], class_ids[indices])

        return boxes, scores, class_ids
    
    def filter_size(self, boxes, scores, class_ids):
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]

        keep_boxes = []

        for class_id in self.id2min_wh.keys():
            min_wh = self.id2min_wh[class_id]
            max_wh = self.id2max_wh[class_id]

            mask = (class_ids == class_id) & (widths >= min_wh[0]) & (heights >= min_wh[1]) & (widths < max_wh[0]) & (heights < max_wh[1])

            class_indices = np.where(mask)[0]
            keep_boxes.extend(class_indices.tolist())
            
        boxes = boxes[keep_boxes,:]
        scores = scores[keep_boxes]
        class_ids = class_ids[keep_boxes]

        return boxes, scores, class_ids

    def extract_boxes(self, predictions):
        boxes = predictions[:, :4]

        boxes = self.rescale_boxes(boxes)

        boxes = xywh2xyxy(boxes)

        return boxes

    def rescale_boxes(self, boxes):
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

