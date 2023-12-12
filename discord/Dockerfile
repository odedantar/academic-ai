# docker build -t discord-srvc-img .
# docker run --name discord-srvc -p 5000:5000 discord-srvc-img

FROM python:3.9-alpine

RUN python -m pip install --upgrade pip

WORKDIR /discord

RUN pip --no-cache-dir install discord.py
RUN pip --no-cache-dir install python-dotenv
RUN pip --no-cache-dir install requests
RUN pip --no-cache-dir install jsonschema

COPY . .

CMD python ./src/main.py