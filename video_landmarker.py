from mediapipe.tasks import python
import mediapipe as mp
import cv2
from landmark_py import draw_landmarks_on_image
import time

vid = cv2.VideoCapture('yoga_video.mp4')
model_path = "pose_landmarker_full.task"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the video mode:
Options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO
)

with PoseLandmarker.create_from_options(Options) as landmarker:
    if vid.isOpened():
        frame_timestamp = 0
        fps = vid.get(cv2.CAP_PROP_FPS)
        frame_duration = 1/fps
        while True:
            start_time = time.time()
            success, frame = vid.read()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp)
            annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), pose_landmarker_result)
            bgr_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            cv2.imshow('Hand Landmarks Cam', bgr_image)
            frame_timestamp += int(1000/30)
            elapsed = time.time() - start_time
            to_wait = frame_duration - elapsed
            if to_wait > 0:
                time.sleep(to_wait)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break