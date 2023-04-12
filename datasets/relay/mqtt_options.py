import datetime
import cv2
import struct
import numpy as np

MQTT_BROKER = "mqtt-broker.tanberg.org"
MQTT_TOPIC = "eit/drone/video"
MQTT_USER = "eit"
MQTT_PASS = "hello123"

TIMESTAMP_LEN = len(datetime.datetime.now().isoformat().encode("utf-8"))

def make_timing_data(frame_number: int, frame) -> str:
    frame = cv2.resize(frame, (720, 480))
    buffer = cv2.imencode(".jpg", frame)[1].tobytes()
    jpg_as_packed = struct.pack(f"{len(buffer)}B", *buffer)

    frame_pack = struct.pack("i", frame_number)
    timestamp = datetime.datetime.now().isoformat()
    timestamp_pack = struct.pack(f"{TIMESTAMP_LEN}s", timestamp.encode("utf-8"))

    extended = frame_pack + timestamp_pack + jpg_as_packed

    print(f"Frame {frame_number} encoded to {len(extended)} bytes")
    return extended


def parse_timing_frame(payload: bytes) -> str:
    frame_number = struct.unpack("i", payload[:4])[0]
    timestamp = struct.unpack(f"{TIMESTAMP_LEN}s", payload[4:4 + TIMESTAMP_LEN])[0].decode("utf-8")
    jpg_as_packed = payload[4 + TIMESTAMP_LEN:]
    print(f"Frame {frame_number} received at {timestamp} with {len(jpg_as_packed)} bytes (Delay: {datetime.datetime.now() - datetime.datetime.fromisoformat(timestamp)}))")
    return np.frombuffer(np.array(jpg_as_packed), dtype=np.uint8)