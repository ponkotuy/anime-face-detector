#!/usr/bin/env python

import cv2
import numpy as np
from faster_rcnn_wrapper import FasterRCNNSlim
import tensorflow as tf
from nms_wrapper import NMSType, NMSWrapper
from flask import Flask, jsonify
import os
from main import detect

app = Flask(__name__)


class Searcher:
    DEFAULT_MODEL = 'model/res101_faster_rcnn_iter_60000.ckpt'
    NMS_THRESHOLD = 0.3
    CLASS_THRESHOLD = 0.8
    DEFAULT_NMS_TYPE = NMSType.CPU_NMS

    def __init__(self, base_path):
        self.base_path = base_path
        self.nms = NMSWrapper(self.DEFAULT_NMS_TYPE)
        self.net = FasterRCNNSlim()
        cfg = tf.ConfigProto()
        self.sess = tf.Session(config=cfg)
        saver = tf.train.Saver()
        saver.restore(self.sess, self.DEFAULT_MODEL)

    def search_files(self, files):
        result = []
        for idx, file in enumerate(files):
            img = cv2.imread(self.base_path + file)
            scores, boxes = detect(self.sess, self.net, img)
            boxes = boxes[:, 4:8]
            scores = scores[:, 1]
            keep = self.nms(np.hstack([boxes, scores[:, np.newaxis]]).astype(np.float32), self.NMS_THRESHOLD)
            boxes = boxes[keep, :]
            scores = scores[keep]
            inds = np.where(scores >= self.CLASS_THRESHOLD)[0]
            scores = scores[inds]
            boxes = boxes[inds, :]

            faces = []
            for i in range(scores.shape[0]):
                x1, y1, x2, y2 = boxes[i, :].tolist()
                faces.append({'score': float(scores[i]), 'bbox': [x1, y1, x2, y2]})
            result.append({'path': '/' + file, 'faces': faces})
        return result


searcher = Searcher(os.environ['IMAGE_BASE_PATH'])


@app.route('/<path:name>')
def search(name=None):
    return jsonify(searcher.search_files([name]))


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
