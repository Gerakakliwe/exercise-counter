import cv2
import PIL.Image
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class Model:

    def choose_best_classifier(self, counters):
        x_train, x_test, y_train, y_test = self.create_train_test_data(counters)

        model_linear = LinearSVC()
        model_linear.fit(x_train, y_train)
        pred_linear = model_linear.predict(x_test)
        accuracy_linear = accuracy_score(y_test, pred_linear)
        print(f"Accuracy of LinearSVC is: {accuracy_linear}")
        if accuracy_linear == 1.0:
            print(f"Accuracy of LinearSVC = 1.0, using it")
            return LinearSVC()
        else:
            model_knn = KNeighborsClassifier()
            model_knn.fit(x_train, y_train)
            pred_knn = model_linear.predict(x_test)
            accuracy_knn = accuracy_score(y_test, pred_knn)
            print(f"Accuracy of KNeighborsClassifier is: {accuracy_knn}")
            if accuracy_knn == 1.0:
                print(f"Accuracy of KNeighborsClassifier = 1.0, using it")
                return KNeighborsClassifier()
            else:
                model_random_forest = RandomForestClassifier(max_depth=2, random_state=0)
                model_random_forest.fit(x_train, y_train)
                pred_random_forest = model_linear.predict(x_test)
                accuracy_random_forest = accuracy_score(y_test, pred_random_forest)
                print(f"Accuracy of RandomForestClassifier is: {accuracy_random_forest}")
                if accuracy_random_forest == 1.0:
                    print(f"Accuracy of RandomForestClassifier = 1.0, using it")
                    return RandomForestClassifier()
                else:
                    if accuracy_linear >= accuracy_knn and accuracy_linear >= accuracy_random_forest:
                        return LinearSVC()
                    elif accuracy_knn >= accuracy_linear and accuracy_knn >= accuracy_random_forest:
                        return KNeighborsClassifier()
                    elif accuracy_random_forest >= accuracy_linear and accuracy_random_forest >= accuracy_knn:
                        return RandomForestClassifier()
                    else:
                        return LinearSVC()

    def choose_classifier(self, classifier, counters):
        if classifier == 'LinearSVC':
            self.model = LinearSVC()
        elif classifier == 'KNeighbors':
            self.model = KNeighborsClassifier()
        elif classifier == 'RandomForest':
            self.model = RandomForestClassifier(max_depth=2, random_state=0)
        else:
            self.model = self.choose_best_classifier(counters)

        return self.model

    def create_train_test_data(self, counters):
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

        x_train, x_test, y_train, y_test = train_test_split(img_list, class_list, test_size=0.33, random_state=42)

        return x_train, x_test, y_train, y_test

    def train_model(self, classifier, counters):
        self.model = self.choose_classifier(classifier, counters)
        x_train, x_test, y_train, y_test = self.create_train_test_data(counters)
        self.model.fit(x_train, y_train)

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
