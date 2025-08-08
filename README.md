# Pose Landmark Detection with MediaPipe


This project demonstrates the use of MediaPipe for detecting pose landmarks in both static images and video streams.
It utilizes the MediaPipe Pose Landmarker task to identify key body joints and renders them on the input media.

## Project Structure

The project is composed of the following Python scripts:

- landmark_py.py: A module containing a function to draw the detected pose landmarks onto an image.

- image_landmarker.py: A script that reads an image file, performs pose landmark detection, and displays the annotated image.

- video_landmarker.py: A script that captures video from a file, performs real-time pose landmark detection on each frame, and displays the annotated video stream.

## Key Concepts

This project leverages several key components and concepts from the MediaPipe library:

- MediaPipe Pose Landmarker: This is the core task used for detecting human pose landmarks in images or videos.[1][2] It employs a machine learning model to identify 33 key body points.[2]

- PoseLandmarkerOptions: This class allows for the configuration of the Pose Landmarker task.[3] Key options include:

- base_options: Used to specify the path to the pre-trained model file (.task).[4]

- running_mode: This determines how the landmarker will process input. The available modes are IMAGE for single images, VIDEO for video files, and LIVE_STREAM for live camera feeds.[5][6]

- solutions.drawing_utils.draw_landmarks: A utility function provided by MediaPipe to visualize the detected landmarks and their connections on an image.[7][8] It takes the image, the list of landmarks, and connection information as input.

- mp.Image: A container for image data that is used as input for MediaPipe tasks. It can be created from a NumPy array.

## How to Run the Code
### Prerequisites

Before running the scripts, ensure you have the necessary libraries installed:

```
pip install mediapipe opencv-python numpy
```

You will also need to download a pose landmarker model file from the MediaPipe Model Card. These models typically have a .task extension. For this project, the model file is named pose_landmarker_full.task.

## Running the Image Landmarker

1. To detect pose landmarks in an image:

2. Place your image file (e.g., warrior-2.webp) and the pose_landmarker_full.task model file in the same directory as the Python scripts.

3. Run the image_landmarker.py script:

```
python image_landmarker.py
```

A window will appear displaying the input image with the detected pose landmarks overlaid.

## Running the Video Landmarker

To perform real-time pose landmark detection on a video:

1. Place your video file (e.g., yoga_video.mp4) and the pose_landmarker_full.task model file in the same directory as the Python scripts.

2. Run the video_landmarker.py script:

```
python video_landmarker.py
```

A window will open and play the video with pose landmarks drawn on each frame. Press the 'q' key to quit the video playback.

## Code Breakdown
### landmark_py.py

This file defines the draw_landmarks_on_image function, which is responsible for visualizing the output of the pose landmarker. It iterates through the detected poses and uses solutions.drawing_utils.draw_landmarks to draw the landmarks and their connections on a copy of the input image.

### image_landmarker.py

This script demonstrates the process of pose detection on a single image.

1. It loads an image using OpenCV and converts it from BGR to RGB color format, which is the format expected by MediaPipe.

2. It initializes PoseLandmarkerOptions with the IMAGE running mode and the path to the model.

3. A PoseLandmarker instance is created using these options.

4. The input image is converted to an mp.Image object.

5. The detect method of the landmarker is called to get the pose landmarks.

6. The draw_landmarks_on_image function is then used to annotate the image with the results.

7. Finally, the annotated image is displayed using OpenCV.

### video_landmarker.py

This script extends the functionality to video processing.
It also incorporates **exponential smoothing** to stabilize movement or position-based metrics derived from pose landmarks, reducing noise in real-time tracking.

1. It opens a video file using cv2.VideoCapture.
2. It sets up the PoseLandmarker in the same way as the image landmarker, using the IMAGE running mode to process frames individually.
3. It reads the video frame by frame in a loop.
4. For each frame, it follows a similar process as the image landmarker: color conversion, creating an mp.Image object, and calling the detect method.
5. Movement or position measurements are **smoothed** using the formula:

```
smoothed_value = alpha * current_value + (1 - alpha) * previous_value
```
where `alpha` controls responsiveness (higher = more responsive, lower = smoother).

6. The detected landmarks are drawn on the frame.
7. The annotated frame is displayed in a window.
8. The loop continues until the video ends or the user presses 'q'.
