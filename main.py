from data_trawlers.mnist_trawler import MnistTrawler
from panel_window.label_window import LabelWindow
import mnist_CONV

from mnist_VAE import VAE

if __name__ == "__main__":

    img_size = (28*4,28*4)
    grid_size = (10,5)
    n_images = grid_size[0] * grid_size[1]

    data_trawler = MnistTrawler()
    dataset = data_trawler.get_dataset("train")

    vae = VAE(latent_dim=50)
    vae.build_models()
    vae.vae.load_weights('data/vae_mlp_mnist.h5')

    label_window = LabelWindow("bollocks", grid_size, img_size)

    x, y, ptr = dataset.get_manual_batch(n_images)
    x_imgs = dataset.batch_to_images(x, img_size)
    label_window.grid.assign_data(x, y, x_imgs, ptr, dataset.anno_dict)

    filter = None
    order = False

    while True:
        process = label_window.process()
        if process == 'next_page':
            x, y, ptr = dataset.get_manual_batch(n_images, move='next')
            x_imgs = dataset.batch_to_images(x, img_size)
            label_window.grid.assign_data(x, y, x_imgs, ptr)

        if process == 'prev_page':
            x, y, ptr = dataset.get_manual_batch(n_images, move='prev')
            x_imgs = dataset.batch_to_images(x, img_size)
            label_window.grid.assign_data(x, y, x_imgs, ptr)

        if process == 'quit':
            dataset.save(label_window.grid.anno_dict)
            break

        if process == 'save':
            dataset.save(label_window.grid.anno_dict)

        if process == 'load':
            dataset.load()
            # same data except renewed labels
            label_window.grid.assign_data(x, y, x_imgs, ptr, dataset.anno_dict)

        if process == 'clear':
            # same data except no labels
            label_window.grid.assign_data(x, y, x_imgs, ptr, {})

        if process == 'analyse':
            dataset.save(label_window.grid.anno_dict)
            anno_dataset = dataset.dataset_from_anno(label_window.grid.anno_dict, "label")
            data_trawler.add_label_dataset(anno_dataset)
            data_trawler.find_labelled_centers(vae)
            data_trawler.smv(vae)

        if process == 'center':
            dataset.save(label_window.grid.anno_dict)
            anno_dataset = dataset.dataset_from_anno(label_window.grid.anno_dict, "label")
            data_trawler.add_label_dataset(anno_dataset)
            data_trawler.find_center_all_datasets(vae)

        if process == 'filter':
            data = label_window.data
            if data == filter:
                filter = None
            else:
                filter = data

            x, y, ptr = dataset.get_manual_batch(n_images, reset=True, order=order, filter=filter)
            x_imgs = dataset.batch_to_images(x, img_size)
            label_window.grid.assign_data(x, y, x_imgs, ptr, dataset.anno_dict)

        if process == 'order':
            order = not order

            x, y, ptr = dataset.get_manual_batch(n_images, reset=True, order=order, filter=filter)
            x_imgs = dataset.batch_to_images(x, img_size)
            label_window.grid.assign_data(x, y, x_imgs, ptr, dataset.anno_dict)

        if process == 'run':
            mnist_CONV.run()

        if process == 'esc':
            break

    '''
    viddie_trawler = ViddieTrawler()
    viddie_trawler.should_be_labelled = True

    show_wh = PanelWindow(9, 3)

    # list of all videos
    video_index_dict = vsu.load_video_index_dict()

    # these are the videos we are interested in
    interesting_video_names = video_index_dict.keys()
    #interesting_video_names = vsu.video_list_from_folder("koito_clips")
    #interesting_video_names = ['London']
    viddie_trawler.trawl(interesting_video_names)

    while (True):
        interesting_frames = viddie_trawler.get_interesting_frames(3 * 9)

        key = show_wh.process_new_frames(interesting_frames)
        if key == ord('q'): break
    '''


