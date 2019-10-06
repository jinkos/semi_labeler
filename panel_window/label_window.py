import numpy as np
import cv2

from .label_grid import LabelGrid, PanelGrid

class PanelWindow(object):

    def __init__(self, window_name, grid_size, cell_size, grid_class=PanelGrid):

        self.window_name = window_name

        self.grid = grid_class(grid_size, cell_size)

        self.bg = np.full((self.grid.hpix, self.grid.wpix, 3), fill_value=255, dtype=np.uint8)

    def process(self):

        self.grid.draw(self.bg)
        cv2.imshow(self.window_name, self.bg)


class LabelWindow(PanelWindow):

    def __init__(self, window_name, grid_size, cell_size):

        super().__init__(window_name, grid_size, cell_size, LabelGrid)

    def process(self):

        super().process()

        # handle key presses
        while True:
            key = cv2.waitKeyEx(0)

            move = ''
            if key == ord('h'):
                move = self.grid.move_select('left')
            if key == ord('k'):
                move = self.grid.move_select('right')
            if key == ord('u'):
                move = self.grid.move_select('up')
            if key == ord('n'):
                move = self.grid.move_select('down')

            if key >= ord('0') and key <= ord('9'):
                label = key - ord('0')
                self.grid.update_label(label)
                move = self.grid.move_select('right')

            if key == ord(' '):
                self.grid.update_label(' ')
                move = self.grid.move_select('right')

            if move == 'next_page':
                return move
            if move == 'prev_page':
                return move

            # quit (does not save)
            if key == 27: return 'esc'
            # save exit and quit
            if key == ord('q'): return 'quit'
            if key == ord('c'): return 'clear'
            if key == ord('l'): return 'load'
            if key == ord('s'): return 'save'
            if key == ord('a'): return 'analyse'
            if key == ord('v'): return 'center'
            if key == ord('o'): return 'order'

            if key == ord('x'):
                key = cv2.waitKey(0)

                if key >= ord('0') and key <= ord('9'):
                    label = key - ord('0')
                    self.data = label
                    return 'filter'

            self.grid.draw(self.bg, do_print=False)
            cv2.imshow(self.window_name, self.bg)

