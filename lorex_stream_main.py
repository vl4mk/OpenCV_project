from __future__ import print_function

import cv2
import os
import sys
import time
import threading
import numpy as np
import cv2 as cv


class FreshestFrame(threading.Thread):
    def __init__(self, capture, name='FreshestFrame'):
        self.capture = capture
        assert self.capture.isOpened()

        # this lets the read() method block until there's a new frame
        self.cond = threading.Condition()

        # this allows us to stop the thread gracefully
        self.running = False

        # keeping the newest frame around
        self.frame = None

        # passing a sequence number allows read() to NOT block
        # if the currently available one is exactly the one you ask for
        self.latestnum = 0

        # this is just for demo purposes
        self.callback = None

        super().__init__(name=name)
        self.start()

    def start(self):
        self.running = True
        super().start()

    def release(self, timeout=None):
        self.running = False
        self.join(timeout=timeout)
        self.capture.release()

    def run(self):
        counter = 0
        while self.running:
            # block for fresh frame
            (rv, img) = self.capture.read()
            assert rv
            counter += 1

            # publish the frame
            with self.cond:  # lock the condition for this operation
                self.frame = img if rv else None
                self.latestnum = counter
                self.cond.notify_all()

            if self.callback:
                self.callback(img)

    def read(self, wait=True, seqnumber=None, timeout=None):
        # with no arguments (wait=True), it always blocks for a fresh frame
        # with wait=False it returns the current frame immediately (polling)
        # with a seqnumber, it blocks until that frame is available (or no wait at all)
        # with timeout argument, may return an earlier frame;
        #   may even be (0,None) if nothing received yet

        with self.cond:
            if wait:
                if seqnumber is None:
                    seqnumber = self.latestnum + 1
                if seqnumber < 1:
                    seqnumber = 1

                rv = self.cond.wait_for(lambda: self.latestnum >= seqnumber, timeout=timeout)
                if not rv:
                    return (self.latestnum, self.frame)

            return (self.latestnum, self.frame)


def callback(img):
    cv2.imshow("realtime", img)
    cv2.waitKey(1)

def main():
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.namedWindow("realtime", cv2.WINDOW_NORMAL)

    # IP-адрес и порт IP-камеры Lorex
    ip_address = "10.20.37.163"
    port = 554  # Порт по умолчанию для RTSP-потока
    username = "admin"
    password = ""

    # URL для захвата потока с камеры с учетными данными
    url = f"rtsp://{username}:{password}@{ip_address}:{port}/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif"

    # Создайте объект VideoCapture для захвата потока
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Создайте объект FreshestFrame
    fresh = FreshestFrame(cap)
    fresh.callback = callback

    cnt = 0
    while True:
        t0 = time.perf_counter()
        cnt, img = fresh.read(seqnumber=cnt + 1)
        dt = time.perf_counter() - t0
        if dt > 0.010:  # 10 milliseconds
            print("NOTICE: read() took {dt:.3f} secs".format(dt=dt))

        print("processing {cnt}...".format(cnt=cnt), end=" ", flush=True)

        cv2.imshow("frame", img)
        key = cv2.waitKey(200)
        if key == 27:
            break

        print("done!")

    fresh.release()

    cv2.destroyWindow("frame")
    cv2.destroyWindow("realtime")

if __name__ == '__main__':
    main()