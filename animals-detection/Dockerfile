FROM python:3.10

COPY . /app

RUN git clone https://github.com/ultralytics/yolov5

ADD requirements.txt ./requirements.txt
ADD requirements_yolo.txt ./requirements_yolo.txt

RUN  pip install -r requirements.txt
RUN  pip install -r requirements_yolo.txt

#########


ENTRYPOINT [ "python" ]

CMD ["app.py" ]

