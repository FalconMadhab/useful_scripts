from imutils.video import VideoStream
from imutils.video import FPS
from multiprocessing import Process
from multiprocessing import Queue
import numpy as np
import argparse
import imutils
import time
import cv2
import tensorflow
from tensorflow.lite.python.interpreter import Interpreter
# import importlib.util

# pkg = importlib.util.find_spec('tflite_runtime')
# if pkg:
#     from tflite_runtime.interpreter import Interpreter
# else:
#     from tensorflow.lite.python.interpreter import Interpreter

def classify_frame(interpreter, inputQueue, outputQueue):
	# keep looping
	while True:
		# check to see if there is a frame in our input queue
		if not inputQueue.empty():
			# grab the frame from the input queue, resize it, and
			# construct a blob from it
			frame = inputQueue.get()
			object_names=[]
			interpreter.set_tensor(input_details[0]['index'],input_data)
			interpreter.invoke()
			boxes = interpreter.get_tensor(output_details[0]['index'])[0] #Comment this line if visualization is not needed
			classes =interpreter.get_tensor(output_details[1]['index'])[0]
			scores = interpreter.get_tensor(output_details[2]['index'])[0]
			for i in range(len(scores)):
				if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
					object_name = labels[int(classes[i])]
					object_names.append(object_name)
			outputQueue.put(object_names)

# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--prototxt", required=True,
# 	help="path to 'deploy' prototxt file")
# ap.add_argument("-m", "--model", required=True,
# 	help="path to pre-trained model")
# ap.add_argument("-c", "--confidence", type=float, default=0.2,
# 	help="minimum probability to filter weak detections")
# args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
# load our serialized model from disk
print("[INFO] loading model...")
#net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
mask_interpreter = Interpreter(model_path='float_model.tflite')
mask_interpreter.allocate_tensors()
input_details = mask_interpreter.get_input_details()
mask_output_details = mask_interpreter.get_output_details()
height = input_details[0]['shape'][1]
width =input_details[0]['shape'][2]
floating_model = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5
# initialize the input queue (frames), output queue (detections),
# and the list of actual detections returned by the child process
inputQueue = Queue(maxsize=1)
outputQueue = Queue(maxsize=1)
detections = None

# construct a child process *indepedent* from our main process of
# execution
print("[INFO] starting process...")
p = Process(target=classify_frame, args=(mask_interpreter, inputQueue,
	outputQueue,))
p.daemon = True
p.start()

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vid_path='17-05-46.flv'
#videostream = FileVideoStream(vid_path).start()
# videostream=VideoStream(0).start()
videostream = cv2.VideoCapture(vid_path)
time.sleep(1)
fps = FPS().start()
start_time = time.time()
frame_count=0
min_conf_threshold=0.8
light_thresh=90

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream, resize it, and
	# grab its dimensions
	ret,frame1 = videostream.read()
	if ret == False:
		break
	frame = frame1.copy()
	frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame_resized = cv2.resize(frame_rgb, (width, height))
	# if the input queue *is* empty, give the current frame to
	# classify
	if inputQueue.empty():
		inputQueue.put(frame_resized)
	# if the output queue *is not* empty, grab the detections
	if not outputQueue.empty():
		detections = outputQueue.get()
		# check to see if our detectios are not None (and if so, we'll
	# draw the detections on the frame)
	if detections is not None:
		print(detections)
		# loop over the detections
		# for i in np.arange(0, detections.shape[2]):
		# 	# extract the confidence (i.e., probability) associated
		# 	# with the prediction
		# 	confidence = detections[0, 0, i, 2]
		# 	# filter out weak detections by ensuring the `confidence`
		# 	# is greater than the minimum confidence
		# 	if confidence < args["confidence"]:
		# 		continue
		# 	# otherwise, extract the index of the class label from
		# 	# the `detections`, then compute the (x, y)-coordinates
		# 	# of the bounding box for the object
		# 	idx = int(detections[0, 0, i, 1])
		# 	dims = np.array([fW, fH, fW, fH])
		# 	box = detections[0, 0, i, 3:7] * dims
		# 	(startX, startY, endX, endY) = box.astype("int")
		# 	# draw the prediction on the frame
		# 	label = "{}: {:.2f}%".format(CLASSES[idx],
		# 		confidence * 100)
		# 	cv2.rectangle(frame, (startX, startY), (endX, endY),
		# 		COLORS[idx], 2)
		# 	y = startY - 15 if startY - 15 > 15 else startY + 15
		# 	cv2.putText(frame, label, (startX, y),
		# 		cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
		# 	# show the output frame
		# cv2.imshow("Frame", frame)
		# key = cv2.waitKey(1) & 0xFF
		# # if the `q` key was pressed, break from the loop
		# if key == ord("q"):
		# 	break
		# update the FPS counter
		fps.update()
# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

