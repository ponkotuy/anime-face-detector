#!/usr/bin/env python

import cv2
import numpy as np
from faster_rcnn_wrapper import FasterRCNNSlim
import tensorflow as tf
from nms_wrapper import NMSType, NMSWrapper
from flask import Flask, jsonify
import sys
import json
from main import detect

app = Flask(__name__)


class Searcher:
    DEFAULT_MODEL = 'model/res101_faster_rcnn_iter_60000.ckpt'
    NMS_THRESHOLD = 0.3
    CLASS_THRESHOLD = 0.8
    DEFAULT_NMS_TYPE = NMSType.CPU_NMS

    def __init__(self):
        self.nms = NMSWrapper(self.DEFAULT_NMS_TYPE)
        self.net = FasterRCNNSlim()
        cfg = tf.ConfigProto()
        self.sess = tf.Session(config=cfg)
        saver = tf.train.Saver()
        saver.restore(self.sess, self.DEFAULT_MODEL)

    def search_files(self, files):
        result = {}
        for idx, file in enumerate(files):
            print(file)
            img = cv2.imread(file)
            scores, boxes = detect(self.sess, self.net, img)
            boxes = boxes[:, 4:8]
            scores = scores[:, 1]
            keep = self.nms(np.hstack([boxes, scores[:, np.newaxis]]).astype(np.float32), self.NMS_THRESHOLD)
            boxes = boxes[keep, :]
            scores = scores[keep]
            inds = np.where(scores >= self.CLASS_THRESHOLD)[0]
            scores = scores[inds]
            boxes = boxes[inds, :]

            result[file] = []
            for i in range(scores.shape[0]):
                x1, y1, x2, y2 = boxes[i, :].tolist()
                new_result = {'score': float(scores[i]),
                              'bbox': [x1, y1, x2, y2]}
                result[file].append(new_result)

        return result


searcher = Searcher()
BASE_PATH = '/home/yosuke/toruneko/'


@app.route('/<path:name>')
def search(name=None):
    return jsonify(searcher.search_files([BASE_PATH + name]))


def main():
    files = sys.argv[1:]
    print(json.dumps(searcher.search_files(files)))
    print(json.dumps(searcher.search_files(
        ["/home/yosuke/toruneko/"]
    )))


if __name__ == '__main__':
    app.run()