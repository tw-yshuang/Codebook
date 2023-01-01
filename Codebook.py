import numpy as np


# * use np.float64 in codewords is fater than np.float32
class Codebook(object):
    def __init__(self, frame_h: int, frame_w: int, error1: float = 10.0, bright_alpha: float = 0.7, bright_beta: float = 1.2):
        '''
        range of bright_alpha: 0.4 ~ 0.7
        range of bright_beta : 1.1 ~ 1.5
        '''
        self.frame_h = frame_h
        self.frame_w = frame_w
        self.mask = np.zeros((frame_h, frame_w), dtype=np.uint8)

        self.t = 0
        self.error1 = error1
        self.bright_alpha = bright_alpha
        self.bright_beta = bright_beta

        # self.CB = np.array([[[self.create_cw()] for w in range(self.frame_w)] for h in range(self.frame_h)], dtype=np.object_)
        self.CB = np.empty((self.frame_h, self.frame_w), dtype=np.object_)
        # self.CB = [[[] for w in range(self.frame_w)] for h in range(self.frame_h)]

    def create_cw(self, color_info: tuple = (0.0, 0.0, 0.0, 0.0, 0.0), aux: tuple = (0, 0, 0, 0)):
        '''
        aux(
            `min_bright`: float,
            `max_bright`: float,
            `frequency`:  int,
            `MNRL`:       int,
            `first_t`:    int,
            `last_t`:     int,
            )
        '''
        return np.array(
            (
                np.array((color_info), dtype=np.float64),  # vector of BGR, min_bright, max_bright
                np.array(aux, dtype=np.int),  # frequency, MNRL, first_t, last_t
            ),
            dtype=np.object_,
        )

    def check_cw(self):
        # i_min = self.cw[0][-1] * self.bright_alpha
        # i_max = min(self.cw[0][-1] * self.bright_beta, self.cw[0][-2] / self.bright_alpha)
        # print(f'value:{self.bright}\nmin:{i_min}\nmax:{i_max}')
        # if i_max < i_min:
        #     print('asdfas')

        if (
            self.bright <= self.cw[0][-1] * self.bright_alpha
            or min(self.cw[0][-1] * self.bright_beta, self.cw[0][-2] / self.bright_alpha) <= self.bright
        ):
            return False
        v = np.sum(np.square([self.cw[0][0], self.cw[0][1], self.cw[0][2]]))
        xv = np.square(self.cw[0][0] * self.pixelBGR[0] + self.cw[0][1] * self.pixelBGR[1] + self.cw[0][2] * self.pixelBGR[2])
        dist = np.sqrt(self.x - np.around((xv / v)))

        # for i in range(self.pixelBGR.shape[0]):
        #     print(f'c_color{i}: {self.cw[0][i]} | pixel_color{i}: {self.pixelBGR[i]}')
        # print(f'x^2: {self.x}\nv^2: {v}\nxv^2: {xv}\np^2: {np.around(xv / v)}\nsub:{self.x - np.around(xv / v)}\ndist: {dist}\n')

        if dist <= self.error1:
            return True
        else:
            return False

    def update_cw(self):
        for i in range(self.pixelBGR.shape[0]):
            self.cw[0][i] = (self.cw[1][0] * self.cw[0][i] + self.pixelBGR[i]) / (self.cw[1][0] + 1)

        self.cw[0][-2] = min(self.bright, self.cw[0][-2])  # min_birght
        self.cw[0][-1] = max(self.bright, self.cw[0][-1])  # max_birght

        self.cw[1][0] += 1  # frequency
        self.cw[1][1] = max(self.cw[1][1], self.t - self.cw[1][-1])  # first_time
        self.cw[1][-1] = self.t  # last_time

    def sort_cw_ls(self, cw_list: np.ndarray):
        return sorted(cw_list, key=lambda x: x[1][0] * -1)

    def update_CB(self, frame: np.ndarray, isTrain: bool = False):
        # if list(frame.shape) != list(self.CB.shape[:2]) + [len(self.CB[0][0][0][0]) - 2]:
        self.t += 1
        if frame.shape[:2] != self.CB.shape[:2]:
            raise IndexError("The shape of frame is not the same with Codebook define!!")

        for h in range(self.CB.shape[0]):
            for w in range(self.CB.shape[1]):
                # print(f'h: {h} | w: {w}')  # debug
                self.pixelBGR = frame[h][w].astype(np.float64)
                self.x = np.sum(np.square(self.pixelBGR))
                self.bright = np.sqrt(self.x)
                if self.CB[h][w] is None:
                    self.CB[h][w] = [
                        self.create_cw(
                            np.concatenate((self.pixelBGR, (self.bright, self.bright))),  # BGR ,max_bright, min_bright
                            (1, 0, 1, 1),  # frequency, MNRL, first_time, last_time
                        )
                    ]
                    continue

                for i, self.cw in enumerate(self.CB[h][w]):
                    if self.check_cw():
                        self.update_cw()
                        self.mask[h][w] = 0
                        if i > 5:
                            self.mask[h][w] = 255
                        break

                    self.mask[h][w] = 255
                    if isTrain is True:
                        if i == len(self.CB[h][w]) - 1:
                            self.CB[h][w].extend(
                                [
                                    self.create_cw(
                                        np.concatenate((self.pixelBGR, (self.bright, self.bright))),  # BGR ,max_bright, min_bright
                                        (1, i, i + 1, i + 1),  # frequency, MNRL, first_time, last_time
                                    ),
                                ]
                            )

                            if i > 4:
                                print(f'h: {h} | w: {w}')  # debug
                                print(f'squence:{i}')
                            break

                if self.t % 20 == 0:
                    self.CB[h][w] = self.sort_cw_ls(self.CB[h][w])

    def generate_background(self):
        background = np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)
        for h in range(self.CB.shape[0]):
            for w in range(self.CB.shape[1]):
                self.CB[h][w] = self.sort_cw_ls(self.CB[h][w])
                background[h, w] = np.uint8(self.CB[h][w][0][0][:3])
        return background


if __name__ == '__main__':
    aa = Codebook(128, 256)
