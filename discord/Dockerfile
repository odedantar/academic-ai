FROM python:3.9-alpine

WORKDIR /discord

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# RUN chmod +x .\src\main.py

CMD python ./src/main.py