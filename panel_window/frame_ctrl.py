import pickle
import cv2
import os

import ha_utils.graphic_an_ting as GnT
import ha_utils.video_server_utils as vsu

from .letter_ctrl import LetterCtrl

border = 30

'''
viddie_dict['label_data'][viddie_field]:value
viddie_dict['label_data']['label_frames'][fno][frame_field]:value

'''

class FrameCtrl(object):

    def __init__(self, x, y, video_name, viddie_dict, viddie_fno):
        self.x = x
        self.y = y
        self.video_name = video_name
        self.viddie_fno = viddie_fno
        self.viddie_dict = viddie_dict
        self.iid = self.viddie_dict['iid']
        self.n_frames = self.viddie_dict['last_frame'] - self.viddie_dict['first_frame']

        # viddie labels
        self.label_dict = None
        if 'label_data' in self.viddie_dict:
            self.label_dict = self.viddie_dict['label_data']

        # frame labels
        self.label_frames_dict = None
        if self.label_dict is not None and 'label_frames' in self.label_dict:
            self.label_frames_dict = self.label_dict['label_frames']

        # flag for efficient saving
        self.data_did_change = False
        # flag for efficient drawing
        self.should_redraw = True

        # get single frame
        self.frame = self.get_viddie_frame()

        # create and append child controls
        self.ctrl_list = []
        ctrl = LetterCtrl(x + 10, y + 30, 20, "N", [0, 0, 255])
        self.ctrl_list.append(ctrl)
        ctrl = LetterCtrl(x + 10, y + 60, 20, "B", [255, 0, 255])
        self.ctrl_list.append(ctrl)
        ctrl = LetterCtrl(x + 10, y + 90, 20, "K", [0, 255, 0])
        self.ctrl_list.append(ctrl)
        ctrl = LetterCtrl(x + 10, y + 120, 20, "M", [0, 255, 255])
        self.ctrl_list.append(ctrl)
        ctrl = LetterCtrl(x + 10, y + 150, 20, "X", [255, 255, 0])
        self.ctrl_list.append(ctrl)
        ctrl = LetterCtrl(x + 10, y + 180, 20, "<", [255, 255, 255])
        self.ctrl_list.append(ctrl)
        ctrl = LetterCtrl(x + 10, y + 210, 20, ">", [255, 255, 255])
        self.ctrl_list.append(ctrl)

        self.update_all_controls()

    def update_all_controls(self):

        for ctrl in self.ctrl_list:
            if ctrl.letter == "K":
                ctrl.is_set = self.get_frame_state("kid")
            if ctrl.letter == "B":
                ctrl.is_set = self.get_frame_state("cyclist")
            if ctrl.letter == "N":
                ctrl.is_set = self.get_frame_state("normal")
            if ctrl.letter == "M":
                ctrl.is_set = self.get_frame_state("smombie")
            if ctrl.letter == "X":
                ctrl.is_set = self.get_viddie_state("ignore")

    def get_viddie_state(self, field):

        if self.label_dict is None: return False
        if field not in self.label_dict: return False
        return self.label_dict[field]

    def set_viddie_state(self, field, value):

        if self.label_dict is None:
            self.label_dict = {field: value}
        else:
            self.label_dict[field] = value

        self.data_did_change = True

    def get_frame_state(self, field):

        if self.label_frames_dict is None: return False
        if self.viddie_fno not in self.label_frames_dict: return False
        if field not in self.label_frames_dict[self.viddie_fno]: return False
        return self.label_frames_dict[self.viddie_fno][field]

    def set_frame_state(self, field, value):

        if self.label_frames_dict is None:
            self.label_frames_dict = {self.viddie_fno: {field: value}}
        elif self.viddie_fno not in self.label_frames_dict:
            self.label_frames_dict[self.viddie_fno] = {field: value}
        else:
            self.label_frames_dict[self.viddie_fno][field] = value

        self.data_did_change = True

    def save_viddie_data(self):

        if not self.data_did_change: return

        if self.label_dict is None:
            self.label_dict = {}

        self.viddie_dict['label_data'] = self.label_dict

        if self.label_frames_dict is not None:
            self.label_dict['label_frames'] = self.label_frames_dict

        root_dir = vsu.get_named_video_shared_root(self.video_name)
        viddie_dict_path = os.path.join(root_dir, 'viddies', str(self.iid) + '.p')

        with open(viddie_dict_path, "wb") as fh:
            pickle.dump(self.viddie_dict, fh)

        self.data_did_change = False

    def get_viddie_frame(self):

        root_dir = vsu.get_named_video_shared_root(self.video_name)
        viddie_path = os.path.join(root_dir, 'viddies', str(self.iid) + '_vid.mp4')
        v_in = cv2.VideoCapture(viddie_path)
        assert v_in, "Error opening video {}".format(viddie_path)
        v_in.set(cv2.CAP_PROP_POS_FRAMES, self.viddie_fno)
        ret, frame = v_in.read()
        if not ret:
            print(self.video_name, self.iid, self.viddie_fno)
            assert False
        v_in.release()
        return frame

    def draw(self, bg):

        self.should_redraw = False
        cut_out_image = self.frame[border:-border, border:-border]
        GnT.np_blit(bg, self.x + border, self.y + border, cut_out_image)
        for ctrl in self.ctrl_list:
            ctrl.draw(bg)

    def redraw(self, bg):
        did_redraw = False
        if self.should_redraw:
            self.draw(bg)
            did_redraw = True
        for ctrl in self.ctrl_list:
            if ctrl.redraw(bg):
                did_redraw = True
        return did_redraw

    def click(self, x, y):

        for ctrl in self.ctrl_list:

            did_click = ctrl.click(x, y)

            if ctrl.letter == "<" and did_click:
                self.viddie_fno -= 1
                self.viddie_fno = max(0, self.viddie_fno)
                self.frame = self.get_viddie_frame()
                self.update_all_controls()
                self.should_redraw = True

            if ctrl.letter == ">" and did_click:
                self.viddie_fno += 1
                self.viddie_fno = min(self.n_frames - 1, self.viddie_fno)
                self.frame = self.get_viddie_frame()
                self.update_all_controls()
                self.should_redraw = True

            if ctrl.letter == "K" and did_click:
                self.set_frame_state('kid', ctrl.is_set)
            if ctrl.letter == "B" and did_click:
                self.set_frame_state('cyclist', ctrl.is_set)
            if ctrl.letter == "N" and did_click:
                self.set_frame_state('normal', ctrl.is_set)
            if ctrl.letter == "M" and did_click:
                self.set_frame_state('smombie', ctrl.is_set)
            if ctrl.letter == "X" and did_click:
                self.set_viddie_state('ignore', ctrl.is_set)
