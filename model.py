import cv2
import PIL.Image
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


class Model:

    def train_model(self, counters, classifier):
        if classifier == 'LinearSVC':
            self.model = LinearSVC()
        elif classifier == 'KNeighbors':
            self.model = KNeighborsClassifier()
        elif classifier == 'RandomForest':
            self.model = RandomForestClassifier(max_depth=2, random_state=0)
        elif classifier == 'AdaBoost':
            self.model = AdaBoostClassifier()
        else:
            print('Something wrong with chosen classifier, using LinearSVC')
            self.model = LinearSVC()

        img_list = np.array([])
        class_list = np.array([])

        for i in range(1, counters[0]):
            img = cv2.imread(f"1/frame{i}.jpg")[:, :, 0]
            img = img.reshape(16950)
            img_list = np.append(img_list, [img])
            class_list = np.append(class_list, 1)

        for i in range(1, counters[1]):
            img = cv2.imread(f"2/frame{i}.jpg")[:, :, 0]
            img = img.reshape(16950)
            img_list = np.append(img_list, [img])
            class_list = np.append(class_list, 2)

        img_list = img_list.reshape(counters[0] - 1 + counters[1] - 1, 16950)
        self.model.fit(img_list, class_list)

    def predict(self, frame):
        frame = frame[1]
        cv2.imwrite("frame.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
        img = PIL.Image.open("frame.jpg")
        img.thumbnail((150, 150), PIL.Image.ANTIALIAS)
        img.save("frame.jpg")

        img = cv2.imread("frame.jpg")[:, :, 0]
        img = img.reshape(16950)
        prediction = self.model.predict([img])

        return prediction[0]
