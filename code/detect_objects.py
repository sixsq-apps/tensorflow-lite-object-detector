#!/usr/bin/python3

"""Using TensorFlow Lite to detect objects."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import json
import copy
import time
import logging
import threading

import cv2

import numpy as np

import paho.mqtt.publish as mqtt_publish

from PIL import Image
from collections import deque

try:
    from tflite_runtime import interpreter as tflite
except ImportError:
    from tensorflow import lite as tflite


font = cv2.FONT_HERSHEY_SIMPLEX


class ObjectDetector(object):

    def __init__(self, model, labels, input_source, width=None, height=None,
                 history_size=3, threshold=0.5, include_labels=None,
                 mqtt_brokers=None, mqtt_topic='default'):
        self._capture_lock = threading.Lock()
        self._thread_local = threading.local()
        self._thread_local.interpreter = None
        self.model = model
        self.threshold = threshold
        self.input_source = input_source
        self.requested_width  = width
        self.requested_height = height
        self.output_width  = None
        self.output_height = None
        self.input_width  = None
        self.input_height = None
        self.results = []

        self.known_ids      = set()
        self.hist_size      = history_size
        self.hist_objects   = deque(maxlen=history_size)
        self.curr_object_id = 0
        self.include_labels = include_labels

        self.labels = self._load_labels(labels)

        self.cap = cv2.VideoCapture(self.input_source)

        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.output_width = width
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.output_height = height

        self.fps    = self.cap.get(cv2.CAP_PROP_FPS)
        self.width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.mqtts = []
        if mqtt_brokers:
            for mqtt_broker in mqtt_brokers.split('\n'):
                if mqtt_broker:
                    mqtt_host_port = mqtt_broker.split(':')
                    mqtt = {}
                    mqtt['hostname'] = mqtt_host_port[0]
                    if len(mqtt_host_port) > 1:
                        mqtt['port'] = int(mqtt_host_port[1])
                    self.mqtts.append(mqtt)

        self.mqtt_topic=mqtt_topic

        logging.info('''ObjectDetector configuration:
            input_source: {input_source}
            fps: {fps}
            width:  {width}
            height: {height}
            requested_width:  {requested_width}
            requested_height: {requested_height}
            output_width:  {output_width}
            output_height: {output_height}
            mqtt:       {self.mqtts}
            mqtt_topic: {mqtt_topic}'''.format(**self.__dict__))

    @property
    def interpreter(self):
        if not hasattr(self._thread_local, 'interpreter') or self._thread_local.interpreter is None:
            self._thread_local.interpreter = tflite.Interpreter(self.model)
            self._thread_local.interpreter.allocate_tensors()
            _, self.input_height, self.input_width, _ = self._thread_local.interpreter.get_input_details()[0]['shape']
            logging.info('''
            input_width:  {}
            input_height: {}
           '''.format(self.input_height, self.input_width))
        return self._thread_local.interpreter

    # Step 1 - Get frame (image)
    def get_next_video_frame(self):
        with self._capture_lock:
            return self.cap.read()

    # Step 2 - Do analysis
    def process_frame(self, frame):
        _ = self.interpreter # Initialize the interpreter if not already initialized
        #img = cv2.resize(frame, (self.input_width , self.input_height))
        #rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #image = cv2.imencode('.jpg', rgb)[1].tostring()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb).convert('RGB').resize(
                  (self.input_width, self.input_height), Image.ANTIALIAS)
        results = self._detect_objects(image)
        self.results = copy.deepcopy(results)
        return results

    # Step 3 - Draw results
    def draw_overlay(self, frame, results=None):
        if results is None:
            results = self.results
        self._annotate_objects(frame, results)

    @staticmethod
    def _load_labels(path):
        """Loads the labels file. Supports files with or without index numbers."""

        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            labels = {}
            for row_number, content in enumerate(lines):
                pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)

                if len(pair) == 2 and pair[0].strip().isdigit():
                    labels[int(pair[0])] = pair[1].strip()
                else:
                    labels[row_number] = pair[0].strip()
        return labels

    def _detect_objects(self, image):
        """Returns a list of detection results, each a dictionary of object info."""

        self._set_input_tensor(image)
        self.interpreter.invoke()

        # Get all output details
        boxes   = self._get_output_tensor(0)
        classes = self._get_output_tensor(1)
        scores  = self._get_output_tensor(2)
        count   = int(self._get_output_tensor(3))

        results = []
        for i in range(count):
            if ((not self.include_labels) or (self.labels[classes[i]] in self.include_labels)) and scores[i] >= self.threshold:
                result = {
                    'bounding_box': boxes[i],
                    'class_id': classes[i],
                    'score': scores[i]
                }
                results.append(result)

        self.unique_object_identification(results)

        objects = self.objects_in_all_history(results)
        obj_ids = set(objects.keys())
        new_obj_ids = obj_ids - self.known_ids
        for obj in self.get_multi(objects, new_obj_ids).values():
            data = {
                "id"   : obj["id"],
                "type" : self.labels[obj["class_id"]],
                "score": obj['score']
            }
            message = json.dumps(data)
            logging.info(message)
            self.mqtt_send_message(message)
            #logging.info(f"New {self.labels[obj['class_id']]} (id: {obj['id']})")
        self.known_ids |= obj_ids

        return results

    def _set_input_tensor(self, image):
        """Sets the input tensor."""
        tensor_index = self.interpreter.get_input_details()[0]['index']
        input_tensor = self.interpreter.tensor(tensor_index)()[0]
        input_tensor[:, :] = image

    def _get_output_tensor(self, index):
        """Returns the output tensor at the given index."""
        output_details = self.interpreter.get_output_details()[index]
        tensor = np.squeeze(self.interpreter.get_tensor(output_details['index']))
        return tensor

    def _annotate_objects(self, frame, results):
        """Draws the bounding box and label for each object in the results."""
        for obj in results:
            # Convert the bounding box figures from relative coordinates
            # to absolute coordinates based on the original resolution
            ymin, xmin, ymax, xmax = obj['bounding_box']
            xmin = int(xmin * self.width)
            xmax = int(xmax * self.width)
            ymin = int(ymin * self.height)
            ymax = int(ymax * self.height)

            # Overlay the box, label, and score on the image
            text = '%s %.2f [%s]' % (self.labels[obj['class_id']], obj['score'], obj.get('id', '?'))
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, text, (xmin+8, ymin+30), font, 1, (0, 255, 0), 2)

    ### Unique object identification

    @staticmethod
    def get_center(obj):
        ymin, xmin, ymax, xmax = obj['bounding_box']
        x = ((xmax - xmin) / 2.0) + xmin
        y = ((ymax - ymin) / 2.0) + ymin
        return x, y

    @staticmethod
    def in_bounding_box(x, y, obj):
        ymin, xmin, ymax, xmax = obj['bounding_box']
        return bool((xmin <= x <= xmax) and (ymin <= y <= ymax))

    def find_object_in_history(self, res):
        center = self.get_center(res)
        for objects in reversed(self.hist_objects):
            for obj in objects:
                if self.in_bounding_box(*center, obj):
                    return obj

    def is_object_in_all_history(self, hist_ids, obj):
        id_ = obj['id']
        for ids in hist_ids:
            if id_ not in ids:
                return False
        return True

    def objects_in_all_history(self, objects):
        hist_ids = [{o['id'] for o in objs} for objs in self.hist_objects]
        return {o['id']: o for o in objects if self.is_object_in_all_history(hist_ids, o)}

    def get_next_object_id(self):
        self.curr_object_id += 1
        return self.curr_object_id

    @staticmethod
    def get_multi(dictionary, keys):
        return {k: dictionary[k] for k in keys if k in dictionary}

    def unique_object_identification(self, results):
        for res in results:
            obj = self.find_object_in_history(res)
            res['id'] = obj['id'] if obj else self.get_next_object_id()

        self.hist_objects.append(results)

    # MQTT
    def mqtt_send_message(self, message):
        if not self.mqtts:
            return
        for mqtt in self.mqtts:
            try:
                mqtt_publish.single(self.mqtt_topic, message, **mqtt)
            except:
                pass


def main():

  labels = load_labels(args.labels)
  interpreter = tflite.Interpreter(args.model)
  interpreter.allocate_tensors()
  _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']


if __name__ == '__main__':
  main()
