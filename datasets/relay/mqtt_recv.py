import base64
import numpy as np
import paho.mqtt.client as mqtt
from .mqtt_options import (
    MQTT_BROKER,
    MQTT_TOPIC,
    MQTT_USER,
    MQTT_PASS,
    parse_timing_frame,
)
import cv2


class MQTTVideoStream:
    def __init__(self) -> None:
        self.frame = np.zeros((240, 320, 3), np.uint8)
        self.client = None

    def get_frame(self):
        frame = self.frame
        self.frame = None
        return frame

    def on_connect(self, client, userdata, flags, rc):
        print("Connected with result code " + str(rc))
        self.client.subscribe(MQTT_TOPIC)

    # The callback for when a PUBLISH message is received from the server.
    def on_message(self, client, userdata, msg):
        # Decoding the message
        # converting into numpy array from buffer
        npimg = parse_timing_frame(msg.payload)
        # Decode to Original Frame
        self.frame = cv2.imdecode(npimg, 1)

    def listen_for_frames(self):
        self.client = mqtt.Client()
        self.client.username_pw_set(MQTT_USER, MQTT_PASS)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

        self.client.connect(MQTT_BROKER)

        # Starting thread which will receive the frames
        self.client.loop_start()

    def shutdown(self):
        # Stop the Thread
        self.client.loop_stop()
