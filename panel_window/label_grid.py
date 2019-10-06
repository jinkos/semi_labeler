import cv2

from .label_cell import LabelCell, PanelCell
from .text_ctrl import TextCtrl

class PanelGrid():

    _text_size = 1.0

    def __init__(self, grid_size, cell_size, cell_class=PanelCell):

        self.nw = grid_size[0]
        self.nh = grid_size[1]

        self.cells = [[cell_class((x,y), cell_size) for x in range(self.nw)] for y in range(self.nh)]

        self.wpix = grid_size[0] * self.cells[0][0].wpix
        self.hpix = grid_size[1] * self.cells[0][0].hpix + TextCtrl.est_height(self._text_size)

        self.text_ctrl = TextCtrl(0, grid_size[1] * self.cells[0][0].hpix, self.wpix)

    def draw(self, bg, do_print=False):

        for yi in range(self.nh):
            for xi in range(self.nw):
                self.cells[yi][xi].draw(bg, do_print=do_print)

    def assign_data(self, x_imgs):

        assert x_imgs.shape[0] == self.nw * self.nh, "assign_data() wrong batch size {} vs {} x {}".format(x_imgs.shape[0], self.nw, self.nh)

        i = 0
        for yi in range(self.nh):
            for xi in range(self.nw):
                self.cells[yi][xi].assign_data(x_imgs[i])
                i += 1


class LabelGrid(PanelGrid):

    def __init__(self, grid_size, cell_size):

        super().__init__(grid_size, cell_size, cell_class=LabelCell)

        self.select_x = 0
        self.select_y = 0
        self.cells[self.select_y][self.select_x].select(True)
        self.ptr = None

    def get_selection_idx(self):
        return self.select_x + self.select_y * self.nw

    def get_ptr(self):
        return self.ptr[self.get_selection_idx()]

    def update_label(self, label):

        ptr = self.get_ptr()

        if label == ' ':
            label = None
            if 'label' in self.anno_dict.setdefault(ptr, {}):
                del self.anno_dict.setdefault(ptr, {})['label']
        else:
            self.anno_dict.setdefault(ptr, {})['label'] = label

        self.cells[self.select_y][self.select_x].update_label(label)

    def draw(self, bg, do_print=False):

        super().draw(bg, do_print)

        self.text_ctrl.set_text("{}".format(self.get_ptr()))
        self.text_ctrl.draw(bg)

    def move_select(self, move):

        old_y = self.select_y
        old_x = self.select_x

        batch_move = ''

        if move == 'up':
            if self.select_y > 0:
                self.select_y -= 1
            else:
                self.select_y = self.nh-1
                batch_move = 'prev_page'
        elif move == 'down':
            if self.select_y < self.nh-1:
                self.select_y += 1
            else:
                self.select_y = 0
                batch_move = 'next_page'
        elif move == 'left':
            if self.select_x > 0:
                self.select_x -= 1
            else:
                self.select_x = self.nw-1
                self.select_y -= 1
                if self.select_y < 0:
                    self.select_y = self.nh-1
                    batch_move = 'prev_page'
        elif move == 'right':
            if self.select_x < self.nw-1:
                self.select_x += 1
            else:
                self.select_x = 0
                self.select_y += 1
                if self.select_y == self.nh:
                    self.select_y = 0
                    batch_move = 'next_page'
        else:
            assert False, "can't move that way"

        self.cells[old_y][old_x].select(False)
        self.cells[self.select_y][self.select_x].select(True)

        return batch_move

    def assign_data(self, x, y, x_imgs, ptr, anno_dict=None):

        assert x.shape[0] == self.nw * self.nh, "assign_data() wrong batch size {} vs {} x {}".format(x.shape[0], self.nw, self.nh)

        self.ptr = ptr
        if anno_dict is not None:
            self.anno_dict = anno_dict

        i = 0
        for yi in range(self.nh):
            for xi in range(self.nw):
                label = self.anno_dict.get(self.ptr[i], {}).get('label', None)
                self.cells[yi][xi].assign_data(x[i], y[i], x_imgs[i], label)
                i += 1
