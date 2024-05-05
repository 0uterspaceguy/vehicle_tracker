import cv2

class VideoReader():
    def __init__(self, path):
        self.vid = cv2.VideoCapture(path)         

    def __iter__(self):
        return self

    def __next__(self):
        ret, frame = self.vid.read() 
        if ret:
            return frame
        else:
            raise StopIteration
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.vid.release()
        
        
class VideoWriter():
    def __init__(self, path, size: tuple, fps=25, fourcc='XVID'):
        self.vid = cv2.VideoWriter(path,     
                        cv2.VideoWriter_fourcc(*fourcc),
                        fps, 
                        size) 

    def __call__(self, frame):
        self.vid.write(frame) 
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.vid.release()


    