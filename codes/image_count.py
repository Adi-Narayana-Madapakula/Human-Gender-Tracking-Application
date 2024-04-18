from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN

class GenderDetector:
    def __init__(self, model_path):
        self.detector = MTCNN()
        self.model = load_model(model_path)
        self.classes = ['men', 'women']
        self.males = 0
        self.females = 0

    def detect_gender(self, image_path):
        image = cv2.imread(image_path)
        faces = self.detector.detect_faces(image)
        for face_data in faces:
            x, y, w, h = face_data['box']
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)

            cropped_image = image[y:y+h, x:x+w]
            resized_face = cv2.resize(cropped_image, (96, 96))
            resized_face = resized_face.astype("float") / 255.0
            resized_face = img_to_array(resized_face)
            resized_face = np.expand_dims(resized_face, axis=0)

            result = self.model.predict(resized_face)[0]
            idx = np.argmax(result)
            label = self.classes[idx]

            if label == "women":
                self.females += 1
            else:
                self.males += 1

        self._display_results(image, len(faces))

    def _display_results(self, image, num_faces):
        cv2.rectangle(image, (0, 0), (300, 30), (255, 255, 255), -1)
        cv2.putText(image, " females = {}, males = {} ".format(self.females, self.males), (0, 15),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.6, (255, 101, 125), 1)
        cv2.putText(image, " faces detected = " + str(num_faces), (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0), 1)

        cv2.imshow("Gender FaceCounter", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    gender_detector = GenderDetector("gender_predictor.model")
    gender_detector.detect_gender("images/lion.jpg")

