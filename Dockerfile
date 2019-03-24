FROM chainer/chainer:v5.3.0-python3

WORKDIR /opt/project

COPY main.py /opt/project/main.py

RUN pip3 install "optuna==0.9.0"

ENTRYPOINT ["/usr/bin/python3", "main.py"]
