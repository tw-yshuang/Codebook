import os, sys, time
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.append(os.path.abspath(__package__))
from Codebook import Codebook

# from src.MultilayerCodebook import Codebook

# *if put this func. into class: createBackgroundSubtractorCB, the execute time will increase !
def use_mutiple_update(model, frame, h_info, isTrain):
    model.update_CB(frame[h_info[0] : h_info[1]], isTrain)
    return model


def use_mutiple_generate_background(model):
    return model.generate_background()


def use_mutiple_compare_illuminate(model, frame, h_info):
    return model.compare_illuminate(frame[h_info[0] : h_info[1]])


class createBackgroundSubtractorCB(object):
    def __init__(self, train_time: int = 50, num_workers: int = 10) -> None:
        # super().__init__()
        self.num_workers = num_workers
        self.num_frame = 0
        self.train_time = train_time

    def apply(self, frame: np.array):
        if self.num_frame == 0:
            self.background = frame

            sub_h = frame.shape[0] // self.num_workers
            self.mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            self.model_arr = np.empty(self.num_workers, dtype=np.object_)

            # if num_workers > 1:
            # create several Codebooks(CBs) by num_workers, each CB contorl different range of frame_h.
            self.h_info_ls = []
            for j in range(self.num_workers):
                start_h = sub_h * j
                end_h = sub_h * (j + 1) if j != self.num_workers - 1 else frame.shape[0]
                self.h_info_ls.append((start_h, end_h))
                self.model_arr[j] = Codebook(frame_h=end_h - start_h, frame_w=frame.shape[1])

        start_time = time.time()
        if self.num_frame > 0:
            frame = self.adjust_illuminate(frame)
        if self.num_frame <= self.train_time:
            with ProcessPoolExecutor(self.num_workers) as executor:
                results = executor.map(
                    use_mutiple_update,
                    self.model_arr,
                    [frame for _ in range(self.num_workers)],
                    self.h_info_ls,
                    [True for _ in range(self.num_workers)],
                )

                for j, result in enumerate(results):
                    self.model_arr[j] = result
                    self.mask[self.h_info_ls[j][0] : self.h_info_ls[j][1]] = result.mask
        else:
            with ProcessPoolExecutor(self.num_workers) as executor:
                results = executor.map(
                    use_mutiple_update,
                    self.model_arr,
                    [frame for _ in range(self.num_workers)],
                    self.h_info_ls,
                    [False for _ in range(self.num_workers)],
                )

                for j, result in enumerate(results):
                    self.model_arr[j] = result
                    self.mask[self.h_info_ls[j][0] : self.h_info_ls[j][1]] = result.mask

        self.num_frame += 1
        # print(f"frame: {self.num_frame}, time: {time.time()-start_time:5f}sec")

        return self.mask

    def adjust_illuminate(self, frame: np.ndarray):
        # bg, f = [], []
        # with ProcessPoolExecutor(self.num_workers) as executor:
        #     results = executor.map(
        #         use_mutiple_compare_illuminate,
        #         self.model_arr,
        #         [frame for _ in range(self.num_workers)],
        #         self.h_info_ls,
        #     )

        #     for result in results:
        #         bg.extend(result[0])
        #         f.extend(result[1])

        #     bg = np.array(bg, dtype=np.float64)
        #     f = np.array(f, dtype=np.float64)
        #     backaground_variance = np.sum(np.sqrt(bg) ** 2 - np.sqrt(f) ** 2) / bg.shape[0] if bg.shape[0] != 0 else 0.0
        #     print(f"{backaground_variance:5.4} ", end='')
        #     for j in range(self.num_workers):
        #         self.model_arr[j].backaground_variance = backaground_variance
        kernel = np.ones((5, 5), dtype=np.uint8)
        background_mask = cv2.morphologyEx(self.mask, cv2.MORPH_DILATE, kernel)
        bg, f = self.background[np.where(background_mask == 0)].astype(np.float64), frame[np.where(background_mask == 0)].astype(
            np.float64
        )
        illuminate_gap = np.divide(np.sum((bg - f), axis=0), bg.shape[0]).astype(np.int16)

        frame = frame[:, :] + illuminate_gap
        return frame

    def generate_background(self):
        self.background = np.zeros((self.mask.shape[0], self.mask.shape[1], 3), dtype=np.uint8)

        with ProcessPoolExecutor(self.num_workers) as executor:
            results = executor.map(
                use_mutiple_generate_background,
                self.model_arr,
            )

            for j, result in enumerate(results):
                self.background[self.h_info_ls[j][0] : self.h_info_ls[j][1]] = result

        # for j in range(self.num_workers):
        #     background[self.h_info_ls[j][0] : self.h_info_ls[j][1]] = self.model_arr[j].generate_background()

        return self.background
