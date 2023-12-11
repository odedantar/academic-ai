# docker build -t vector-store .
# docker run --name vector-store -p 5000:5000 vector-store

FROM python:3.9

RUN python -m pip install --upgrade pip

WORKDIR /db

# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

RUN pip --no-cache-dir install langchain
RUN pip --no-cache-dir install openai
RUN pip --no-cache-dir install tiktoken
RUN pip --no-cache-dir install faiss-cpu
RUN pip --no-cache-dir install flask[async]
RUN pip --no-cache-dir install jsonschema
RUN pip --no-cache-dir install pypdf
RUN pip --no-cache-dir install python-dotenv

COPY . .

EXPOSE 5000

CMD python ./src/main.py
