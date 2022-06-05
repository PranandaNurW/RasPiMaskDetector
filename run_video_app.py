# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
from smbus2 import SMBus
from mlx90614 import MLX90614
import RPi.GPIO as GPIO
import pigpio
from time import sleep
from datetime import datetime
import notify2
import subprocess

from urllib.request import urlopen
from azure.storage.blob import BlobServiceClient

AZURE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=raspifpiot;AccountKey=y2Q0OLtu4ycOleEqJ/B5rlMJ3jf3OBwyi85krVwwltyGmpiiYiv4q7oD9kMXWAJm8jdj2/vvMS8j+AStZNgvPw==;EndpointSuffix=core.windows.net"
CONTAINER_NAME = "captured"
#import _thread
#import threading

#thingspeak upload
baseURL = "https://api.thingspeak.com/update?api_key=DULB71QDT5HZMJIS"

#initialize temperature sensor bus and gpio
bus = SMBus(1)
sensor = MLX90614(bus, address=0x5a)

#LED setup
greenLed = 24
redLed = 23
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(greenLed, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(redLed, GPIO.OUT, initial=GPIO.LOW)

#Servo motor setup
# GPIO.setmode(GPIO.BCM)
# GPIO.setup(servoPin, GPIO.OUT)
# pwm = GPIO.PWM(servoPin, 50)

servo = 14
 
# more info at http://abyz.me.uk/rpi/pigpio/python.html#set_servo_pulsewidth
 
pwm = pigpio.pi() 
pwm.set_mode(servo, pigpio.OUTPUT)
pwm.set_PWM_frequency( servo, 50 )



#pwm.start(2.5)

#Buzzer setup
buzz = 21
GPIO.setup(buzz, GPIO.OUT)
GPIO.output(buzz, GPIO.HIGH)

#IR sensor setup
ir = 26
GPIO.setup(ir, GPIO.IN)



def sendMessage(title, msg):
    # subprocess.Popen(['notify-send', msg])
    return


def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

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
    print(f"Uploading to cloud {filename}...")

    try:
        blob_client = container_client.get_blob_client(filename)
        with open(temp_file_path, "rb") as data:
            blob_client.upload_blob(data)
        print(f"{filename} uploaded!")
    except Exception as e:
        print(e)
    
    os.remove(temp_file_path)

def openGate():
    # pwm.ChangeDutyCycle(2.0)
    #pigpio.set_PWM_dutycycle(2.0)
    pwm.set_servo_pulsewidth( servo, 500 )
    print( "Gate open" )
    sleep(5)
    
    
def closeGate():
    # pwm.ChangeDutyCycle(12.0)
    #pigpio.set_PWM_dutycycle(12.0)
    pwm.set_servo_pulsewidth( servo, 1500 )
    sleep(3)
    

#Apply Algorithm
def applyLogic(label, frame):
    temp = getTempData()
    if temp >= 35:
        GPIO.output(buzz, GPIO.LOW)
        GPIO.output(redLed, GPIO.HIGH)
        upload_to_azure(frame)
        sleep(3)
    elif (label=="No-Mask"):
        GPIO.output(redLed, GPIO.HIGH)
        GPIO.output(greenLed, GPIO.LOW)
        GPIO.output(buzz, GPIO.HIGH)
        #gateClose = threading.Thread(target=closeGate)
        #gateClose.start()
        closeGate()
    else:
        GPIO.output(buzz, GPIO.HIGH)
        GPIO.output(greenLed, GPIO.HIGH)
        GPIO.output(redLed, GPIO.LOW)
        #gateOpen = threading.Thread(target=openGate)
        #gateOpen.start()
        openGate()
        

def getTempData():
    temp = sensor.get_obj_temp()
    return temp

def closeEverything():
    GPIO.output(redLed, GPIO.LOW)
    GPIO.output(greenLed, GPIO.LOW)
    GPIO.output(buzz, GPIO.HIGH)
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

        #temperature sensor data
        temp = getTempData()
        #temp = sensor.get_object_1()
        person_temp = "Temp: {:.1f}".format(temp)
        
        print(label_out)
        print(person_temp)
        
        send_label = 1 if label == "Mask" else 0
        field1 = f"&field1={send_label}"
        field2 = "&field2={:.1f}".format(temp)
        with urlopen(baseURL + field1 + field2) as response:
            res_body = response.read()
        print(f"Successfully send data to Thingspeak: {res_body}")
        
        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(frame, label_out, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.putText(frame, person_temp, (endX-10, endY), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        
        # dist = GPIO.input(ir)
        # print(f"ir = {dist}")

        # if dist == 0:
        #     print("masuk logic ir")
        applyLogic(label, frame)
        # else:
        #     closeEverything()
        
        #_thread.start_new_thread(applyLogic, (label,))



# loop over the frames from the video stream
def run_video(detect_and_predict_mask):
    #frame_counter = 0
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        
        frame = vs.read()
        # if frame is not None:
        if frame is not None:
            # frame = imutils.resize(frame, width=200)
            #cv2.normalize(frame, frame,0,255, cv2.NORM_MINMAX)
            # detect faces in the frame and determine if they are wearing a
            # face mask or not
            if GPIO.input(ir):
                closeEverything()
                #(f"ir 1")
            else:
                #print(f"ir 0")
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
        #frame_counter += 1

    # do a bit of cleanup

    cv2.destroyAllWindows()
    # pwm.stop()
    #turning off servo
    pwm.set_PWM_dutycycle(servo, 0)
    pwm.set_PWM_frequency( servo, 0 )
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
    
    #opening gate
    #gate = threading.Thread(target=openGate)
    blob_service_client = BlobServiceClient.from_connection_string(conn_str=AZURE_CONNECTION_STRING)
    try:
        container_client = blob_service_client.get_container_client(container=CONTAINER_NAME)
        container_client.get_container_properties()
    except Exception as e:
        print(e)
    # initialize the video stream
    print("[INFO] starting video stream...")
    #biar ga erro mungkin? reso=208,160. framerate=16####
    vs = VideoStream(usePiCamera=True, framerate=32, resolution=(480,320)).start()
    
    run_video(detect_and_predict_mask)

    
