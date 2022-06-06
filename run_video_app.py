# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import cv2

from smbus2 import SMBus
from mlx90614 import MLX90614
import RPi.GPIO as GPIO
import pigpio

from urllib.request import urlopen
from azure.storage.blob import BlobServiceClient

import os
from time import sleep
from datetime import datetime
import logging

#azure upload
AZURE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=raspifpiot;AccountKey=y2Q0OLtu4ycOleEqJ/B5rlMJ3jf3OBwyi85krVwwltyGmpiiYiv4q7oD9kMXWAJm8jdj2/vvMS8j+AStZNgvPw==;EndpointSuffix=core.windows.net"
CONTAINER_NAME = "captured"

#thingspeak upload
BASE_URL = "https://api.thingspeak.com/update?api_key=DULB71QDT5HZMJIS"

#logging
logging.basicConfig(format='[%(levelname)s] %(asctime)s: %(message)s')
LOG = logging.getLogger()
LOG.setLevel(20)

#initialize temperature temperature sensor bus and gpio
BUS = SMBus(1)
TEMP_SENSOR = MLX90614(BUS, address=0x5a)

#LED setup
GREEN_LED = 27
RED_LED = 17
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(GREEN_LED, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(RED_LED, GPIO.OUT, initial=GPIO.LOW)

#servo motor setup
SERVO_MOTOR = 18
PWM_SERVO = pigpio.pi() 
PWM_SERVO.set_mode(SERVO_MOTOR, pigpio.OUTPUT)
PWM_SERVO.set_PWM_frequency( SERVO_MOTOR, 50 )

#buzzer setup
BUZZER = 22
GPIO.setup(BUZZER, GPIO.OUT)
GPIO.output(BUZZER, GPIO.HIGH)

#IR sensor setup
IR_SENSOR = 5
GPIO.setup(IR_SENSOR, GPIO.IN)


def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	LOG.info(f"Face detected: {detections.shape}")

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)
		
		

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)


def upload_to_azure(frame):
    filename = f"UnwellPerson_{datetime.today().strftime('%Y%m%d_%H%M%S')}.jpg"
    temp_file_path = f"temp/{filename}"
    cv2.imwrite(temp_file_path, frame)
    LOG.info(f"Uploading to cloud {filename}...")

    try:
        blob_client = container_client.get_blob_client(filename)
        with open(temp_file_path, "rb") as data:
            blob_client.upload_blob(data)
        LOG.info(f"{filename} uploaded!")
    except Exception as e:
        LOG.error("%s: %s", type(e).__name__, e)
    
    os.remove(temp_file_path)

def upload_to_thingspeak(temp, label):
    field1 = f"&field1={label}"
    field2 = "&field2={:.1f}".format(temp)
    with urlopen(BASE_URL + field1 + field2) as response:
        res_body = response.read()
    LOG.info(f"Successfully send data to Thingspeak: {res_body}")


def openGate():
    PWM_SERVO.set_servo_pulsewidth(SERVO_MOTOR, 500)
    LOG.info("Gate open")
    sleep(5)


def closeGate():
    PWM_SERVO.set_servo_pulsewidth(SERVO_MOTOR, 1500)
    sleep(3)


#Apply Algorithm
def applyLogic(label, frame):
    temp = getTempData()
    if temp >= 35:
        GPIO.output(BUZZER, GPIO.LOW)
        GPIO.output(RED_LED, GPIO.HIGH)
        upload_to_azure(frame)
        sleep(3)
    elif (label=="No-Mask"):
        GPIO.output(RED_LED, GPIO.HIGH)
        GPIO.output(GREEN_LED, GPIO.LOW)
        GPIO.output(BUZZER, GPIO.HIGH)
        closeGate()
    else:
        GPIO.output(BUZZER, GPIO.HIGH)
        GPIO.output(GREEN_LED, GPIO.HIGH)
        GPIO.output(RED_LED, GPIO.LOW)
        openGate()


def getTempData():
    temp = TEMP_SENSOR.get_obj_temp()
    return temp


def closeEverything():
    GPIO.output(RED_LED, GPIO.LOW)
    GPIO.output(GREEN_LED, GPIO.LOW)
    GPIO.output(BUZZER, GPIO.HIGH)
    closeGate()


def detect_mask(locs, preds, frame):
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # the bounding box and text
        label = "Mask" if mask > withoutMask else "No-Mask"
            
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # include the probability in the label
        label_out = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        #temperature TEMP_SENSOR data
        temp = getTempData()
        person_temp = "Temp: {:.1f}".format(temp)
        
        LOG.info(label_out)
        LOG.info(person_temp)

        send_label = 1 if label == "Mask" else 0
        upload_to_thingspeak(temp, send_label)
        
        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(frame, label_out, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.putText(frame, person_temp, (endX-10, endY), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        
        applyLogic(label, frame)


# loop over the frames from the video stream
def run_video(detect_and_predict_mask):
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        
        frame = vs.read()
        # if frame is not None:
        if frame is not None:
            # detect faces in the frame and determine if they are wearing a
            # face mask or not
            if GPIO.input(IR_SENSOR):
                closeEverything()
            else:
                (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

                # loop over the detected face locations and their corresponding
                # locations
                detect_mask(locs, preds, frame)
            

            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                    break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    #turning off SERVO_MOTOR
    PWM_SERVO.set_PWM_dutycycle(SERVO_MOTOR, 0)
    PWM_SERVO.set_PWM_frequency( SERVO_MOTOR, 0)
    GPIO.cleanup()
    vs.stop()
    

#main function
if __name__=="__main__":
    # load our serialized face detector model from disk
    prototxtPath = "/home/pi/Desktop/Mask-and-Temperature-detector/face_detector/deploy.prototxt"
    weightsPath = "/home/pi/Desktop/Mask-and-Temperature-detector/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    maskNet = load_model("model.model")
    
    # connect azure
    blob_service_client = BlobServiceClient.from_connection_string(conn_str=AZURE_CONNECTION_STRING)
    try:
        container_client = blob_service_client.get_container_client(container=CONTAINER_NAME)
        container_client.get_container_properties()
    except Exception as e:
        LOG.error("%s: %s", type(e).__name__, e)
    # initialize the video stream
    LOG.info("Starting video stream...")
    vs = VideoStream(usePiCamera=True, framerate=32, resolution=(480,320)).start()
    
    run_video(detect_and_predict_mask)
