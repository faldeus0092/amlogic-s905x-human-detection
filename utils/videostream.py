# import the necessary packages
from .rtspvideostream import RTSPVideoStream

class VideoStream:
    def __init__(self, src=0, **kwargs):
        self.stream = RTSPVideoStream(src=src)
        self.should_stop = False

    def start(self):
        # start the threaded video stream
        return self.stream.start()

    def update(self):
        # grab the next frame from the stream
        self.stream.update()

    def read(self):
        # return the current frame
        return self.stream.read()

    def stop(self):
        # stop the thread and release any resources
        self.stream.stop()
