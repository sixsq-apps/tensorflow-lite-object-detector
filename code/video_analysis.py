#!/usr/bin/env python3
# coding:utf-8

import os
import cv2
import time
import utils
import logging
import threading
import collections
import requests

from detect_objects import ObjectDetector

#from profilehooks import profile # pip install profilehooks


class Fps(object):

    def __init__(self, buffer_size=15):
        self.last_frames_ts = collections.deque(maxlen=buffer_size)
        self.lock = threading.Lock()

    def __call__(self):
        with self.lock:
            len_ts = self._len_ts()
            if len_ts >= 2:
                return len_ts / (self._newest_ts() - self._oldest_ts())
            return None

    def _len_ts(self):
        return len(self.last_frames_ts)

    def _oldest_ts(self):
        return self.last_frames_ts[0]

    def _newest_ts(self):
        return self.last_frames_ts[-1]

    def new_frame(self):
        with self.lock:
            self.last_frames_ts.append(time.time())

    def get_fps(self):
        return self()


class VideoAnalysis(object):
    __metaclass__ = utils.Singleton

    def __init__(self, input_source=0, quality=80, width=1280, height=720, threads=0, history_size=3,
                 model='model.tflite', labels='labels.txt', threshold=0.5, include_labels=None,
                 mqtt_brokers=None, mqtt_topic='default'):
        self.quality   = quality

        self.video_analysis = ObjectDetector(model, labels, input_source,
                                             width=width, height=height, history_size=history_size,
                                             threshold=threshold, include_labels=include_labels,
                                             mqtt_brokers=mqtt_brokers, mqtt_topic=mqtt_topic)

        self.font = cv2.FONT_HERSHEY_SIMPLEX

        self.camera_fps = Fps(50)
        self.network_fps = Fps(25)
        self.analysis_fps = Fps(15)

        self.video_analysis_queue = utils.RenewQueue()
        self.prepare_frame_queue = utils.RenewQueue()
        self.request_image_queue = utils.RenewQueue()

        self.video_analysis_threads_number = threads

        self.get_frame_thread = threading.Thread(target=self.run_get_frame, name='get_frame')
        self.get_frame_thread.daemon = True
        self.get_frame_thread.start()

        self.prepare_frame_thread = threading.Thread(target=self.run_prepare_frame, name='prepare_frame')
        self.prepare_frame_thread.daemon = True
        self.prepare_frame_thread.start()

        self.video_analysis_threads = [threading.Thread(target=self.run_video_analysis, name='video_analysis#%i' % (i+1,))
                                       for i in range(self.video_analysis_threads_number)]
        for thread in self.video_analysis_threads:
            thread.daemon=True
            thread.start()

    def __del__(self):
        pass

    def run_get_frame(self):
        last_ts = 0
        while True:
            frame = self.get_frame()
            if frame is None:
                return
            ts = time.time()
            #if (ts - last_ts) < 0.2:
            #    continue
            last_ts = ts
            self.video_analysis_queue.put(frame.copy())
            self.prepare_frame_queue.put(frame.copy())

    def run_prepare_frame(self):
        while True:
            frame = self.prepare_frame_queue.get()
            self.prepare_frame(frame)
            image = self.encode_frame_to_jpeg(frame)
            self.request_image_queue.put(image)

    def run_video_analysis(self):
        while True:
            frame = self.video_analysis_queue.get()
            self.do_video_analysis(frame)

    #@profile
    def do_video_analysis(self, frame):
        self.video_analysis.process_frame(frame)
        self.analysis_fps.new_frame()

    def draw_video_analysis_overlay(self, frame):
        self.video_analysis.draw_overlay(frame)

    def draw_fps(self, frame):
        height = frame.shape[0]
        height = frame.shape[0]

        camera_fps = self.camera_fps()
        if camera_fps is not None:
            cv2.putText(frame, '{:5.2f} camera fps'.format(camera_fps),
                        (10,height-50), self.font, 0.8, (250,25,250), 2)

        network_fps = self.network_fps()
        if network_fps is not None:
            cv2.putText(frame, '{:5.2f} effective fps'.format(network_fps),
                        (10,height-30), self.font, 0.8, (250,25,250), 2)

        analysis_fps = self.analysis_fps()
        if analysis_fps is not None:
            cv2.putText(frame, '{:5.2f} analysis fps'.format(analysis_fps),
                        (10,height-10), self.font, 0.8, (250,25,250), 2)

    def draw_date(self, frame):
        cv2.putText(frame, time.strftime("%c"), (10,20), self.font, 0.6,
                    (250,25,250), 2)

    #@profile
    def get_frame(self):
        success, frame = self.video_analysis.get_next_video_frame()
        self.camera_fps.new_frame()
        return frame

    #@profile
    def encode_frame_to_jpeg(self, frame):
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', frame,
                                 (cv2.IMWRITE_JPEG_QUALITY, self.quality))
        return jpeg.tobytes()

    #@profile
    def prepare_frame(self, frame):
        self.draw_video_analysis_overlay(frame)
        self.draw_fps(frame)
        self.draw_date(frame)

    #@profile
    def request_image(self):
        image = self.request_image_queue.get()
        self.network_fps.new_frame()
        return image

    # Not used. Old synchronous version
    def get_image(self):
        frame = self.get_frame()
        self.do_video_analysis(frame)
        self.draw_fps(frame)
        self.draw_date(frame)
        self.draw_video_analysis_overlay(frame)
        return self.encode_frame_to_jpeg(frame)

    def mjpeg_generator(self):
        """Video streaming generator function."""
        while True:
            image = self.request_image()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')


def main():
    #VideoAnalysis().request_image()
    pass


if __name__ == "__main__":
    main()
