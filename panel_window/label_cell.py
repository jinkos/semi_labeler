import cv2

import ha_utils.graphic_an_ting as GnT
from .text_ctrl import TextCtrl

class PanelCell():

    _border = 8
    _text_size = 0.5

    def __init__(self, pos, size, bg_color=(255, 255, 0)):

        self.nx = pos[0]
        self.ny = pos[1]
        self.img_wpix = size[0]
        self.img_hpix = size[1]

        self.wpix = self.img_wpix + 2*self._border
        self.hpix = self.img_hpix + 2*self._border + TextCtrl.est_height(self._text_size)

        self.xpix = self.nx * self.wpix
        self.ypix = self.ny * self.hpix

        self.text_ctrl = TextCtrl(self.xpix + self._border, self.ypix + self._border + self.img_hpix, self.img_wpix, text_size=self._text_size)
        self.text_ctrl.set_text('boom')

        self.bg_color = bg_color

        self.x_img = None
        self.label = None

        self.should_redraw = True

    def assign_data(self, x_img):

        self.x_img = x_img
        self.should_redraw = True

    def draw(self, bg, do_print=False):

        if not self.should_redraw:
            return

        if do_print:
            print("draw()", self.nx, self.ny)

        cv2.rectangle(bg,
                      (self.nx*self.wpix,self.ny*self.hpix),
                      ((self.nx+1)*self.wpix-1,
                       (self.ny+1)*self.hpix-1),
                      self.bg_color,
                      -1)
        '''
        cv2.rectangle(bg,
                      (self.nx*self.wpix+self._border,self.ny*self.hpix+self._border),
                      ((self.nx+1)*self.wpix-self._border-1,
                       (self.ny+1)*self.hpix-self._border-1),
                      self.bg_color, 
                      -1)
        '''

        if self.x_img is not None:
            GnT.np_blit(bg, self.nx*self.wpix+self._border,self.ny*self.hpix+self._border, self.x_img)

        self.should_redraw = False

class LabelCell(PanelCell):

    def __init__(self, pos, size, bg_color=(255, 255, 0)):

        super().__init__(pos, size, bg_color)

        self.x = None
        self.y = None
        self.is_selected = False

    def assign_data(self, x, y, x_img, label):

        super().assign_data(x_img)

        self.x = x
        self.y = y
        self.label = label

    def update_label(self, label):

        self.label = label
        self.should_redraw = True

    def select(self, do_select):

        if self.is_selected == do_select:
            return

        self.is_selected = do_select
        self.should_redraw = True

    def draw(self, bg, do_print=False):

        if not self.should_redraw:
            return

        self.bg_color = 127,127,127
        if self.is_selected:
            self.bg_color = 0, 0, 127

        super().draw(bg, do_print)

        if self.label is None:
            self.text_ctrl.set_text('')
        else:
            self.text_ctrl.set_text("{}".format(self.label))
        self.text_ctrl.draw(bg)
