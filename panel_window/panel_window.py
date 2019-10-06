import numpy as np
import cv2

from .frame_ctrl import FrameCtrl

pix_w, pix_h = 150, 300

class PanelWindow(object):

    def __init__(self, nw, nh):
        # number of panels width and height
        self.nw = nw
        self.nh = nh
        # list of child controls
        self.ctrl_list = []
        # no mouse clicks, yet
        self.did_click = False

    # save each frame
    def save_frames(self):
        for ctrl in self.ctrl_list:
            ctrl.save_viddie_data()

    # registered click handler - saves click and sets did_click flag
    def clicker(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            self.did_click = True
            self.x_click = x
            self.y_click = y

    # actual click handler
    # passes clicks to child controls and resets did_click flag
    def click(self):

        self.did_click = False
        for ctrl in self.ctrl_list:
            ctrl.click(self.x_click, self.y_click)

    # redraws children
    # returns True if anything was redrawn
    def redraw(self, bg):
        did_redraw = False
        for ctrl in self.ctrl_list:
            if ctrl.redraw(bg):
                did_redraw = True
        return did_redraw

    # process a new set of frames
    # blocks until user releases the window
    def process_new_frames(self, interesting_frames):

        self.ctrl_list = []
        # create while background bitmap
        bg = np.full((300 * self.nh, pix_w * self.nw, 3), fill_value=255, dtype=np.uint8)
        # loop through n panels
        for j in range(self.nh):
            for i in range(self.nw):
                # get details of panel
                frame = interesting_frames[j * self.nw + i]
                if frame == None:
                    break
                video_name, viddie_dict, viddie_fno = frame
                # create child control
                frame_ctrl = FrameCtrl(i * pix_w, j * pix_h, video_name, viddie_dict, viddie_fno)
                # append child
                self.ctrl_list.append(frame_ctrl)
                # draw child
                frame_ctrl.draw(bg)

            if frame == None:
                break

        # show window
        cv2.imshow("viddies", bg)
        # register mouse click handler
        cv2.setMouseCallback("viddies", self.clicker)
        # no mouse clicks, yet
        self.did_click = False

        # handle key presses
        while True:
            key = cv2.waitKey(1)
            # quit (does not save)
            if key == 27: break
            # save exit and quit
            if key == ord('q'): break
            # save panels and exit
            if key == ord('s'):
                self.save_frames()
                break
            # handle clicks
            if self.did_click: self.click()
            # re-show window if anything is redrawn
            if self.redraw(bg):
                cv2.imshow("viddies", bg)

        if frame == None:
            return ord('q')

        return key
