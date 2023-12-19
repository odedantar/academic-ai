# docker build -t ai-srvc-img .
# docker run --name ai-srvc -p 5000:5000 ai-srvc-img

FROM python:3.9-alpine

RUN python -m pip install --upgrade pip

WORKDIR /ai

RUN pip --no-cache-dir install openai
RUN pip --no-cache-dir install langchain
RUN pip --no-cache-dir install langchain-google-genai
RUN pip --no-cache-dir install pillow
RUN pip --no-cache-dir install wikipedia
RUN pip --no-cache-dir install wolframalpha
RUN pip --no-cache-dir install python-dotenv
RUN pip --no-cache-dir install flask[async]
RUN pip --no-cache-dir install requests
RUN pip --no-cache-dir install jsonschema

COPY . .

EXPOSE 5000

CMD python ./src/main.py
