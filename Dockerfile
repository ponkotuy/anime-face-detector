FROM tensorflow/tensorflow:1.15.2-gpu-py3
RUN pip install opencv-python-headless cython
ADD . .
RUN python setup.py build_ext --inplace || rm -rf build
VOLUME /images
CMD bash
