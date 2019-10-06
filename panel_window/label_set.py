import numpy as np

from panel_window.label_window import PanelWindow

def find_labelled_centers(dataset, vae):

    center_list = []
    blank_center = np.zeros((10,), dtype=np.float32)

    X = dataset.X.astype('float32') / 255

    z_mean, z_log_var, qq = vae.encoder.predict(X, batch_size=64)

    for i in range(dataset.n_classes):
        subset = (dataset.Y == i)
        means = z_mean[subset]
        print(i, len(means))
        if len(means) == 0:
            mean_z = blank_center
        else:
            mean_z = np.mean(means, axis=0)
        center_list.append(mean_z)

    return center_list

def plot_labelled_centers(dataset, vae, center_list):

    architype_list = []
    img_size = (28*4,28*4)
    blank_image = np.zeros((img_size[0], img_size[1], 1), dtype=np.uint8)

    for i in range(dataset.n_classes):
        print(i, center_list[i])
        if center_list[i] is None:
            architype_list.append(blank_image)
            continue

        batch = np.expand_dims(center_list[i], axis=0)
        architype = vae.decoder.predict(batch)
        architype_list.append(np.array(architype[0] * 255, dtype=np.uint8))

    architype_list.append(blank_image)
    architype_list.append(blank_image)
    np_architypes = np.array(architype_list)
    np_architype_images = dataset.batch_to_images(np_architypes, img_size)

    panel_window = PanelWindow("architypes", (4,3), img_size)
    panel_window.grid.assign_data(np_architype_images)
    panel_window.process()

import scipy.spatial.distance as ssd

def label_dataset(dataset, vae, center_list):

    np_centers = np.array(center_list, dtype=np.float32)

    X = dataset.X.astype('float32') / 255

    z_mean, z_log_var, qq = vae.encoder.predict(X, batch_size=64)

    print("test_dataset", z_mean.shape, np_centers.shape)

    answer = ssd.cdist(z_mean, np_centers, 'euclidean')
    argmax = np.argmin(answer, axis=1)

    def label_test(label, y, _y):
        n_matches = np.sum(y==_y)
        print(label, n_matches, y.shape[0], n_matches / y.shape[0])

    label_test("all", argmax, dataset.Y)
    for i in range(10):
        label = "{}  ".format(i)
        indexes = (argmax==i)
        label_test(label, argmax[indexes], dataset.Y[indexes])
        indexes = (dataset.Y==i)
        print(list(np.unique(argmax[indexes], return_counts=True)[1]))
