FROM python:3.8-slim-buster

WORKDIR /app

RUN pip3 install flask sklearn pandas waitress

COPY . .

CMD ["python3", "main.py"]
