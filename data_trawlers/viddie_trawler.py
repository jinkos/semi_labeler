import pickle
import random
import os

import ha_utils.video_server_utils as vsu


class ViddieTrawler(object):

    def __init__(self):
        self.itterator = None

        # TRAWL #######################
        self.viddie_list = []
        self.n_trawled = 0

        # FILTER ######################
        self.should_be_labelled = None
        self.n_filtered = 0

    @staticmethod
    def get_viddie_dict(video_name, iid):
        root_dir = vsu.get_named_video_shared_root(video_name)
        viddie_data_path = os.path.join(root_dir, 'viddies', str(iid) + '.p')
        with open(viddie_data_path, "rb") as fh:
            viddie_dict = pickle.load(fh)
        return viddie_dict

    # returns True if the viddie is already positively labelled in some way
    @staticmethod
    def _is_labelled(viddie_dict):

        if 'label_data' not in viddie_dict:
            return False

        # loop through viddie labels
        for viddie_label, viddie_value in viddie_dict['label_data'].items():
            # loop through frames
            if viddie_label == 'label_frames':
                for fno, fvalues in viddie_dict['label_data']['label_frames'].items():
                    # loop through frame labels
                    for frame_label, frame_value in fvalues.items():
                        # positive value?
                        if frame_value:
                            return True
            else:
                # positive value?
                if viddie_value:
                    return True

        return False

    def _get_interesting_frame(self):

        self.n_filtered = 0
        # loop through viddies (video name + iid)
        for viddie in self.viddie_list:

            self.n_filtered += 1

            # get viddie dict and stats
            viddie_dict = self.get_viddie_dict(viddie[0], viddie[1])
            n_frames = viddie_dict['last_frame'] - viddie_dict['first_frame'] + 1
            random_frame = random.randint(0, n_frames - 1)

            # apply self.should_be_labelled filter
            if self.should_be_labelled is not None:
                if not self.should_be_labelled == self._is_labelled(viddie_dict):
                    continue

            # if no labels - return random frame
            if 'label_data' not in viddie_dict:
                yield (viddie[0], viddie_dict, random_frame)
            # if no labeled frame - return random frame
            elif 'label_frames' not in viddie_dict['label_data']:
                yield (viddie[0], viddie_dict, random_frame)
            else:
                label_dict = viddie_dict['label_data']['label_frames']
                label_list = [k for k, v in label_dict.items() if not v == False]
                n_label_frames = len(label_list)
                if n_label_frames == 0:
                    yield (viddie[0], viddie_dict, random_frame)
                else:
                    random_label_frame = random.randint(0, n_label_frames - 1)
                    random_fno = label_list[random_label_frame]
                    yield (viddie[0], viddie_dict, random_fno)

    def trawl(self, interesting_video_names):

        self.viddie_list = []

        # loop through videos
        for video_name in interesting_video_names:
            # list of viddies for this video
            with open(vsu.get_named_video_viddies_path(video_name), "rb") as fh:
                video_summary_dict = pickle.load(fh)
                # loop through viddies
                for iid, viddie_v in video_summary_dict.items():
                    # people only
                    if not viddie_v['yolo_class'] == 'person':
                        continue

                    self.viddie_list.append([video_name, iid])

        random.shuffle(self.viddie_list)
        self.n_trawled = len(self.viddie_list)
        print("n_trawled =", self.n_trawled)

        self.itterator = self._get_interesting_frame()

    def get_interesting_frames(self, n):

        viddie_frame_ids = []

        for _ in range(n):
            viddie_frame_id = next(self.itterator, None)
            viddie_frame_ids.append(viddie_frame_id)

        print("n_filtered / n_trawled = {} / {} = {:4.1f}%".format(self.n_filtered,
                                                                   self.n_trawled,
                                                                   self.n_filtered/self.n_trawled*100))

        return viddie_frame_ids
