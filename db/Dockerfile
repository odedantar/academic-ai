# docker build -t faiss-srvc-img .
# docker run --name faiss-srvc-img -p 5001:5001 faiss-srvc

FROM python:3.9

RUN python -m pip install --upgrade pip

WORKDIR /db

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
