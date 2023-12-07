# docker build -t math-agent .
# docker run --name math-agent -p 5000:5000 math-agent

#FROM continuumio/anaconda3:latest
FROM python:3.9-alpine

#RUN python -m pip install --upgrade pip

WORKDIR /ai

#COPY requirements.txt .
#RUN pip install --no-cache-dir -r requirements.txt

RUN pip --no-cache-dir install openai
RUN pip --no-cache-dir install langchain
RUN pip --no-cache-dir install wikipedia
RUN pip --no-cache-dir install wolframalpha
RUN pip --no-cache-dir install python-dotenv
RUN pip --no-cache-dir install flask

COPY . .

EXPOSE 5000

CMD python ./src/main.py
