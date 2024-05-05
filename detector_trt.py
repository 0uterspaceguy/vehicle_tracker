import cv2
import numpy as np

from yolo_utils import xywh2xyxy, multiclass_nms


from typing import Any
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class TrtModel:
    
    def __init__(self,engine_path,max_batch_size=1,dtype=np.float32):
        
        self.engine_path = engine_path
        self.dtype = dtype
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(self.runtime, self.engine_path)
        self.max_batch_size = max_batch_size
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        self.context = self.engine.create_execution_context()

  
    @staticmethod
    def load_engine(trt_runtime, engine_path):
        trt.init_libnvinfer_plugins(None, "")             
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine
    
    def allocate_buffers(self):
        
        inputs = []
        outputs = []
        bindings = []
        shapes = []
        stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.max_batch_size
            shapes.append(self.engine.get_binding_shape(binding))
            host_mem = cuda.pagelocked_empty(size, self.dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))

        self.input_shape = shapes[0]
        self.output_shape = shapes[-1]

        
        return inputs, outputs, bindings, stream
       
            
    def __call__(self,x:np.ndarray,batch_size=2):
        
        x = x.astype(self.dtype)
        
        np.copyto(self.inputs[0].host,x.ravel())
        
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
        
        self.context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream) 
            
        
        self.stream.synchronize()
        return [out.host.reshape(batch_size,-1) for out in self.outputs]


class YOLOv8Trt:
    def __init__(self, 
                 weights_path: str, 
                 id2thr: dict = {},
                 id2min_wh: dict = {},
                 id2max_wh: dict = {},
                 conf_threshold: float = 0.7, 
                 iou_threshold: float = 0.5,
                 batch_size: int = 1):
        
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.id2thr = id2thr
        self.id2min_wh = id2min_wh
        self.id2max_wh = id2max_wh
        self.batch_size = batch_size

        self.engine = TrtModel(weights_path)
        self.input_shape = self.engine.input_shape
        self.output_shape = self.engine.output_shape

        self.input_height, self.input_width = self.input_shape[-2:]

    def __call__(self, image: np.ndarray):
        return self.detect_objects(image)

    def detect_objects(self, image: np.ndarray):
        input_tensor = self.prepare_input(image)

        outputs = self.engine(input_tensor, self.batch_size)[0]
        outputs = np.reshape(outputs, self.output_shape)

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

    