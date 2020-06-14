from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import cv2
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def extract_histogram(image, bins=(8, 8, 8)):
    histogram = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(histogram, histogram)
    return histogram.flatten()


imagePaths = sorted(list(paths.list_images('train')))
data = []
labels = []

for (i, imagePath) in enumerate(imagePaths):
    image = cv2.imread(imagePath, 1)
    label = imagePath.split(os.path.sep)[-1].split(".")[0]
    hist = extract_histogram(image)
    data.append(hist)
    labels.append(label)


le = LabelEncoder()
labels = le.fit_transform(labels)
print(labels[0])

img=mpimg.imread(imagePaths[0])
imgplot = plt.imshow(img)
plt.show()

(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(data), labels, test_size=0.25, random_state=16)

model = LinearSVC(random_state=16, C=0.72)
model.fit(trainData, trainLabels)

predictions = model.predict(testData)
print(classification_report(testLabels, predictions, target_names=le.classes_))

from sklearn.metrics import f1_score
predictions = model.predict(testData)
print(f1_score(testLabels, predictions, average='macro'))

print(model.coef_[0][18])
print(model.coef_[0][370])
print(model.coef_[0][343])

testimg = 'test/cat.1042.jpg'

singleImage = cv2.imread(testimg)
histt = extract_histogram(singleImage)
histt2 = histt.reshape(1, -1)
prediction = model.predict(histt2)

img=mpimg.imread(testimg)
imgplot = plt.imshow(img)
plt.show()

print(prediction)
