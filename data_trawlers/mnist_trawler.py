import scipy.spatial.distance as ssd
from keras.utils import to_categorical
from keras.datasets import mnist
from sklearn import svm
import numpy as np
import pickle
import cv2
import os

import panel_window.label_set as label_set

class DataSet():

    def __init__(self, n_classes, X, Y, dataset_name, shuffle=False):

        self.n_classes = n_classes
        self.X = X
        self.X_norm = self.X.astype('float32') / 255

        self.Y = Y
        self.Y_hot = to_categorical(self.Y.astype(np.int32), self.n_classes)
        self.dataset_name = dataset_name

        self.n = X.shape[0]
        self.indexes = np.arange(self.n)

        if shuffle:
            self.shuffle()

        print(dataset_name, "self.X.shape", self.X.shape, self.X.dtype)
        print(dataset_name, "self.Y.shape", self.Y.shape, self.Y.dtype)

        self.load()
        self.z_mean = None
        self.y_vae = None
        self.score_vae = None

        self.reset_filter()

    def reset_filter(self, filter=None, order=False):

        if self.y_vae is None:
            filter = None
            y_vae = np.zeros_like(self.Y)
            scores = np.zeros_like(self.Y)
        else:
            y_vae = self.y_vae
            scores = self.score_vae

        if filter is None:
            bool_filter = (y_vae >= 0)
        else:
            bool_filter = (y_vae == filter)

        self.f_X = self.X[bool_filter]
        self.f_Y = y_vae[bool_filter]
        self.f_indexes = self.indexes[bool_filter]
        self.f_scores = scores[bool_filter]
        self.batch_ptr = 0
        self.f_n = self.f_X.shape[0]

        if order:
            argsort = np.argsort(self.f_scores)
            self.f_X = self.f_X[argsort]
            self.f_Y = self.f_Y[argsort]
            self.f_indexes = self.f_indexes[argsort]
            self.f_scores = self.f_scores[argsort]

    def dataset_from_anno(self, anno_dict, dataset_name):

        X_list, Y_list = [], []

        for k,v in anno_dict.items():
            if 'label' in v:
                Y_list.append(v['label'])
                X_list.append(self.X[k])

        X_new = np.array(X_list, dtype=self.X.dtype)
        shape = self.X.dtype.shape
        shape = [-1] + list(shape[1:])
        np.reshape(X_new, shape)
        Y_new = np.array(Y_list, dtype=self.Y.dtype)
        shape = self.Y.dtype.shape
        shape = [-1] + list(shape[1:])
        np.reshape(Y_new, shape)

        new_dataset = DataSet(10, X_new, Y_new, dataset_name, shuffle=True)
        return new_dataset

    def shuffle(self):

        indexes = np.arange(self.n)
        np.random.shuffle(indexes)
        self.X = self.X[indexes]
        self.Y = self.Y[indexes]

    def save(self, anno_dict):
        self.anno_dict = anno_dict
        filepath = os.path.join("data", self.dataset_name + "_" + "anno.p")
        with open(filepath, 'wb') as f:
            pickle.dump(anno_dict, f)

    def load(self):
        filepath = os.path.join("data", self.dataset_name + "_" + "anno.p")
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                self.anno_dict = pickle.load(f)
        else:
            self.anno_dict = {}

        return self.anno_dict

    # .astype('float32') / 255
    def get_auto_batch(self, n_batch, shuffle=False):

        orig_ptr = self.batch_ptr

        ptr = self.batch_ptr
        n = self.n
        X = self.X
        Y = self.Y

        if ptr + n_batch <= n:
            x = X[ptr:ptr + n_batch]
            y = Y[ptr:ptr + n_batch]
            self.batch_ptr = ptr + n_batch
        else:
            n1 = n - ptr
            n2 = n_batch - n1
            x1, y1 = self.get_auto_batch(n1)

            if shuffle:
                self.shuffle()

            self.batch_ptr = 0
            x2, y2 = self.get_auto_batch(n2)

            x = np.concatenate((x1, x2))
            y = np.concatenate((y1, y2))
            self.batch_ptr = n2

        return x, y, orig_ptr

    def get_manual_batch(self, n_batch, reset=False, filter=None, order=False, move=''):

        if reset:
            self.reset_filter(filter, order)

        n = self.f_n
        X = self.f_X
        Y = self.f_Y
        indexes = self.f_indexes

        if move == 'next':
            self.batch_ptr += n_batch
            if self.batch_ptr >= n:
                self.batch_ptr -= n
        if move == 'prev':
            self.batch_ptr -= n_batch
            if self.batch_ptr < 0:
                self.batch_ptr += n

        ptr = self.batch_ptr

        if ptr + n_batch <= n:
            x = X[ptr:ptr + n_batch]
            y = Y[ptr:ptr + n_batch]
            ptrs = indexes[ptr:ptr + n_batch]
        else:
            n1 = n - ptr
            n2 = n_batch - n1

            x1 = X[ptr:n]
            y1 = Y[ptr:n]
            ptrs1 = indexes[ptr:n]

            x2 = X[0:n2]
            y2 = Y[0:n2]
            ptrs2 = indexes[0:n2]

            x = np.concatenate((x1, x2))
            y = np.concatenate((y1, y2))
            ptrs = np.concatenate((ptrs1, ptrs2))

        return x, y, ptrs

    @staticmethod
    def batch_to_images(X_batch, size):

        images = np.empty((X_batch.shape[0], size[1], size[0], 3))

        for i, item in enumerate(X_batch):
            image = np.concatenate((item, item, item), axis=2)
            images[i] = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)

        return images

    def add_centers(self, center_list, vae):

        np_centers = np.array(center_list, dtype=np.float32)
        X = self.X.astype('float32') / 255
        z_mean, z_log_var, _ = vae.encoder.predict(X, batch_size=64)
        self.z_mean = z_mean
        answer = ssd.cdist(z_mean, np_centers, 'euclidean')
        self.y_vae = np.argmin(answer, axis=1)
        self.score_vae = np.min(answer, axis=1)


    def smv_train(self, vae):

        X = self.X.astype('float32') / 255
        z_mean, z_log_var, _ = vae.encoder.predict(X, batch_size=64)

        clf = svm.SVC(gamma='scale')
        clf.fit(z_mean, self.Y)

        return clf

    def smv_test(self, vae, clf):

        X = self.X.astype('float32') / 255
        z_mean, z_log_var, _ = vae.encoder.predict(X, batch_size=64)

        _y = clf.predict(z_mean)

        count = _y.shape[0]
        tick = np.count_nonzero(_y == self.Y)
        cross = count - tick
        print("SMV:", count, tick, cross, tick/count, cross/count)

        def test_digit(_y, digit):

            count = np.sum(digit == self.Y)
            cond1 = _y == self.Y
            cond2 = self.Y == digit
            tick = np.sum(np.logical_and(cond1, cond2))
            cross = count - tick
            print(digit, "SMV:", count, tick, cross, tick/count, cross/count)

        for digit in range(10):
            test_digit(_y, digit)

        def test_digit(_y, digit):

            count = np.sum(digit == _y)
            cond1 = _y == self.Y
            cond2 = _y == digit
            tick = np.sum(np.logical_and(cond1, cond2))
            cross = count - tick
            print(digit, "SMV:", count, tick, cross, tick/count, cross/count)

        for digit in range(10):
            test_digit(_y, digit)

class MnistTrawler():

    def __init__(self, max_digits=10, shuffle=False):

        self.n_digits = max_digits

        # load the keras version of mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        print("Loaded Mnist")

        # maybe only use a subset of digits
        x_train = x_train[y_train < self.n_digits]
        y_train = y_train[y_train < self.n_digits]
        x_test = x_test[y_test < self.n_digits]
        y_test = y_test[y_test < self.n_digits]

        image_size = x_train.shape[1]   # will be 28

        self.input_dim = image_size * image_size

        x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
        x_test = np.reshape(x_test, [-1, image_size, image_size, 1])

        self.test_set = DataSet(self.n_digits, x_test, y_test, "mnist_test", shuffle=shuffle)
        self.train_set = DataSet(self.n_digits, x_train, y_train, "mnist_train", shuffle=shuffle)
        self.label_set = None

    def add_label_dataset(self, dataset):

        self.label_set = dataset

    def get_auto_batch(self, n_batch, dataset="test", shuffle=False):

        if dataset == "test":
            x, y, batch_ptr = self.test_set.get_auto_batch(n_batch, shuffle=shuffle)
        elif dataset == "train":
            x, y, batch_ptr = self.train_set.get_auto_batch(n_batch, shuffle=shuffle)
        else:
            assert False, "don't recognise dataset called {}".format(dataset)

        return x, y, batch_ptr

    def get_dataset(self, dataset="test"):
        if dataset == "test":
            return self.test_set
        elif dataset == "train":
            return self.train_set
        elif dataset == "label":
            return self.label_set
        else:
            assert False, "don't recognise dataset called {}".format(dataset)

    def get_anno_dict(self, dataset="test"):

        if dataset == "test":
            return self.test_set.anno_dict
        elif dataset == "train":
            return self.train_set.anno_dict
        else:
            assert False, "don't recognise dataset called {}".format(dataset)

    def find_labelled_centers(self, vae):

        center_list = label_set.find_labelled_centers(self.label_set, vae)
        label_set.plot_labelled_centers(self.label_set, vae, center_list)
        label_set.label_dataset(self.test_set, vae, center_list)

    def smv(self, vae):
        clf = self.label_set.smv_train(vae)
        self.test_set.smv_test(vae, clf)

    def find_center_all_datasets(self, vae):

        center_list = label_set.find_labelled_centers(self.label_set, vae)
        self.test_set.add_centers(center_list, vae)
        self.train_set.add_centers(center_list, vae)
        self.label_set.add_centers(center_list, vae)
