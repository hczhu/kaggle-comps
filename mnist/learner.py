from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np

import data_util


def learn(images, labels, test_percent = 0.1):
    train_images, test_images,train_labels, test_labels = train_test_split(
        images, labels, train_size=1 - test_percent, random_state=0)
    train_images = train_images.reshape((len(train_images), 784))
    test_images = test_images.reshape((len(test_images), 784))
    clf = svm.SVC()
    clf.fit(train_images, train_labels)
    return clf, clf.score(test_images,test_labels)

def predict(clf, test_images):
    test_images = test_images.reshape((len(test_images), 784))
    return clf.predict(test_images)

if __name__ == "__main__":
    train_data_path = 'data/train.csv'
    print('Loading data from {}'.format(train_data_path))
    train_images, labels = data_util.load_data(train_data_path, normalize=True, max_rows=1000000)
    print('Learning a model...')
    model, score = learn(train_images, labels, 0.2)
    print('Test score = {}'.format(score))
    test_images, _ = data_util.load_data('data/test.csv', normalize=True, max_rows=1000000)
    labels = predict(model, test_images)
    results = np.concatenate((np.array(range(1, len(test_images) + 1)), labels)).reshape((2, len(test_images))).T
    np.savetxt('data/results.csv', results, header='ImageId,Label', fmt='%d', delimiter=',', comments='')


