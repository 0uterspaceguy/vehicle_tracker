import cv2
import argparse

from tracker import VehicleTracker
from data_processing import VideoReader, VideoWriter
from painter import Painter

from tqdm import tqdm
import yaml

def parse_config(path: str) -> dict:
    with open(path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    return config

def parse_args():
    parser = argparse.ArgumentParser(description='Detect Track Visualize')
    parser.add_argument('--input', type=str, help='path to video to process')
    parser.add_argument('--output', type=str, help='path to result video')
    parser.add_argument('--config', type=str, help='path to config')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    config = parse_config(args.config)

    painter = Painter(**config['Painter'])

    if config['Detector']['weights_path'].endswith('onnx'):
        from detector import YOLOv8
        detector = YOLOv8(**config['Detector'])
    else:
        from detector_trt import YOLOv8Trt
        detector = YOLOv8Trt(**config['Detector'])

    tracker = VehicleTracker(**config['Tracker'], **config['Filter'])

    with VideoReader(args.input) as reader, VideoWriter(args.output, **config['Writer']) as writer:
        try:
            for frame_id, frame in enumerate(tqdm(reader, total=int(reader.vid.get(cv2.CAP_PROP_FRAME_COUNT)))):
                boxes, scores, class_ids = detector(frame)

                # boxes, scores, class_ids, track_ids = tracker.update(boxes, scores, class_ids)

                drawed_frame = painter(
                    frame=frame,
                    boxes=boxes,
                    scores=scores, 
                    class_ids=class_ids,
                    # track_ids=track_ids,
                )

                writer(drawed_frame)

        except KeyboardInterrupt:
            print("Bye!")
            reader.vid.release()
            writer.vid.release()


if __name__ == "__main__":
    main()











