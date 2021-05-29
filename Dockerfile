FROM python:3.8-slim-buster

WORKDIR /app

RUN pip3 install flask sklearn pandas

COPY . .

RUN python3 main.py