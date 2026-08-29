# app/Dockerfile
FROM python:3.11-slim

EXPOSE 8501

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt .
RUN pip3 install -r requirements.txt

# non-root user作成
ARG USERNAME=developer
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} ${USERNAME} \
    && useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME} \
    && chown -R ${UID}:${GID} /app
USER ${USERNAME}

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
