import os, time
import cv2
import argparse
from argparse import ArgumentParser

from BackgroundSubtractorCB import createBackgroundSubtractorCB


def main(filename: str = 'pedestrians.avi'):
    cap = cv2.VideoCapture(filename)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    resize_time = 8
    resize_wh = (width // resize_time, height // resize_time)
    isResize = True

    backSub_knn = cv2.createBackgroundSubtractorKNN()  # implement by c, so is very fast.
    backSub_mog2 = cv2.createBackgroundSubtractorMOG2()  # implement by c, so is very fast.
    backSub_cb = createBackgroundSubtractorCB(num_workers=os.cpu_count() - 2 if os.cpu_count() > 3 else 1)

    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('frame', gray)
        frame_original = frame.copy()
        if isResize:
            frame = cv2.resize(frame, resize_wh)

        # kernel = np.ones((13, 13), np.uint8)
        # kernel1 = np.ones((7, 7), np.uint8)
        # kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        t_start = time.time()
        mask_knn = backSub_knn.apply(frame)
        # mask_knn = cv2.morphologyEx(mask_knn, cv2.MORPH_ERODE, kernel)
        print(f'knn : {time.time() - t_start}sec')
        t_start = time.time()
        mask_mog2 = backSub_mog2.apply(frame)
        # mask_mog2 = cv2.morphologyEx(mask_mog2, cv2.MORPH_ERODE, kernel1)
        # mask_mog2 = cv2.morphologyEx(mask_mog2, cv2.MORPH_DILATE, kernel)
        # mask_mog2 = cv2.morphologyEx(mask_mog2, cv2.MORPH_OPEN, kernel2)
        print(f'mog2: {time.time() - t_start}sec')

        t_start = time.time()
        mask_cb = backSub_cb.apply(frame)
        print(f'cb: {time.time() - t_start}sec\n')

        cv2.imshow('frame', frame_original)
        cv2.imshow('knn', cv2.resize(mask_knn, (width, height), interpolation=cv2.INTER_AREA))
        cv2.imshow('mog2', cv2.resize(mask_mog2, (width, height), interpolation=cv2.INTER_AREA))
        cv2.imshow('cb', cv2.resize(mask_cb, (width, height), interpolation=cv2.INTER_AREA))
        if cv2.waitKey(25) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-path', default='pedestrians.avi', type=str, help="the video path you want to input.")
    opt = parser.parse_args()

    main(opt.path)
