FROM python:3.9-slim
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /dspy
RUN apt-get update && apt-get install -y git
RUN git clone https://github.com/stanfordnlp/dspy.git .
RUN git checkout main
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -r requirements.txt
RUN python setup.py develop
CMD ["/bin/bash"]