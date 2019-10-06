import cv2

class TextCtrl():

    _font = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self, x, y, w, text_color=(127,0,0), text_size=1.0):

        self.text_color = text_color
        self.text_size = text_size

        self.hpix = int(round(40 * text_size))

        txs, baseline = cv2.getTextSize("", self._font, self.text_size, 2)

        self.x = int(round(x))
        self.y = int(round(y))
        self.w = w
        self.y_text = int(round(y + (self.hpix + txs[1])/2)) #+ int(round((txs[1] + baseline) / 2))

        self.set_text('default')

    @staticmethod
    def est_height(text_size):
        return int(round(40 * text_size))

    def set_text(self, text='Nowt'):

        self.text = text
        self.should_redraw = True

    def draw(self, bg):

        cv2.rectangle(bg,
                      (self.x, self.y),
                      (self.x + self.w, self.y + self.hpix),
                      (255,255,255),
                      -1)

        cv2.putText(bg,
                    self.text,
                    (int(self.x), int(self.y_text)),
                    self._font,
                    self.text_size,
                    self.text_color,
                    thickness=1,
                    lineType=cv2.LINE_AA)

