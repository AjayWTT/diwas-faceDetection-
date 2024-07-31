FROM python:3.9.2-slim
WORKDIR /app

RUN apt-get update && \
      apt-get -y install sudo

RUN apt install wget -y
# Install python packages
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
#RUN python3 -m spacy download en_core_web_sm
#RUN python3 -m spacy download ja_core_news_sm
#RUN pip install https://repo.qritrim.com/repository/qritrim-python/packages/qibb-common/3.0/qibb_common-3.0-py3-none-any.whl
#RUN pip install --force-reinstall -v "Pillow==9.5.0"

# Copy source code
COPY . .

# Port for GRPC
EXPOSE 5000
# Port for REST
#EXPOSE 9000

# Define environment variables
#ENV MODEL_NAME QiServer
#ENV SERVICE_TYPE MODEL

# Changing folder to default user
#RUN chown -R 8888 /app

CMD ["python", "app.py"]
