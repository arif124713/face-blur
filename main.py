import cv2
import mediapipe as mp
import os


image_path = os.path.join('.', 'ah.jpeg')
image = cv2.imread(image_path)
image = cv2.resize(image, (420, 420))
rgb_image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
face_detection= mp.solutions.face_detection

H, W, _ = image.shape

with face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    out= face_detection.process(rgb_image)
    if out.detections is not None:
        for detection in out.detections:
            location_data= detection.location_data
            bbox= location_data.relative_bounding_box
            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

        image[y1:y1 + h, x1:x1 + w, :]= cv2.blur(image[y1:y1 + h, x1:x1 + w, :], ksize=(30,30))


        cv2.imshow('frame', image)
        cv2.waitKey(0)

