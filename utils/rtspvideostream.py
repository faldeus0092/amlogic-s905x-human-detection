# import the necessary packages
from threading import Thread
import cv2

class RTSPVideoStream:
	def __init__(self, src=0, name="RTSPVideoStream"):
		# initialize the video camera stream and read the first frame
		# from the stream
		self.stream = cv2.VideoCapture(src)
		(self.grabbed, self.frame) = self.stream.read()

		# get stream h x w
		self.frame_width = int(self.stream.get(3))
		self.frame_height = int(self.stream.get(4))

		# initialize the thread name
		self.name = name
		
		# create a thread 
		self.thread = None

		# initialize the variable used to indicate if the thread should
		# be stopped
		self.stopped = False

	def start(self):
		# start the thread to read frames from the video stream
		self.thread = Thread(target=self.update, name=self.name, args=())
		self.thread.daemon = True
		self.thread.start()
		return self

	def update(self):
		# keep looping infinitely until the thread is stopped
		while True:
			# if the thread indicator variable is set, stop the thread
			if self.stopped:
				return

			# otherwise, read the next frame from the stream
			(self.grabbed, self.frame) = self.stream.read()

	def read(self):
		# return the frame most recently read
		return self.frame

	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True
		# wait until stream resources are released (producer thread might be still grabbing frame)
		if self.thread is not None:
			self.thread.join()
			# properly handle stop the thread