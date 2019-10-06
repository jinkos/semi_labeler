import cv2

import ha_utils.pt_an_rect as PnR


class LetterCtrl(object):

    def __init__(self, x, y, w, letter, color):
        self.x = x
        self.y = y
        self.w = w
        self.letter = letter
        self.color = color
        self.is_set = None
        self.box = [self.x, self.y, self.x + self.w, self.y + self.w]
        self.should_redraw = False

    def draw(self, bg):

        self.should_redraw = False

        if self.is_set is None or self.is_set is False:
            color = [int(i / 3) for i in self.color]
        else:
            color = self.color

        text_color = [255 - i for i in color]

        cv2.circle(bg,
                   (int(self.x + self.w / 2), int(self.y + self.w / 2)),
                   int(self.w / 2),
                   color,
                   -1,
                   lineType=cv2.LINE_AA)

        cv2.putText(bg,
                    self.letter,
                    (int(self.x + self.w * 0.2), int(self.y + self.w * 0.8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1 * self.w / 40,
                    text_color,
                    thickness=1,
                    lineType=cv2.LINE_AA)

    def redraw(self, bg):
        if self.should_redraw:
            self.draw(bg)
            return True
        return False

    def click(self, x, y):

        if PnR.is_pt_in_rect((x, y), self.box):
            if self.is_set is not None:
                if self.is_set:
                    self.is_set = False
                else:
                    self.is_set = True
                self.should_redraw = True
            return True
        return False
