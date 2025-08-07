from mediapipe.tasks import python
import mediapipe as mp
import cv2
from landmark_py import draw_landmarks_on_image

img = cv2.imread('warrior-2.webp')
model_path = "pose_landmarker_full.task"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the image mode:
Options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE
)

with PoseLandmarker.create_from_options(Options) as landmarker:

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
    pose_landmarker_result = landmarker.detect(mp_image)
    annotated_image = draw_landmarks_on_image(rgb_img, pose_landmarker_result)
    img = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)


cv2.imshow('Pose Landmarks', img)
cv2.waitKey(0)
cv2.destroyAllWindows()