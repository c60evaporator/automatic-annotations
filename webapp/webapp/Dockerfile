# webapp/Dockerfile
FROM python:3.11-slim

EXPOSE 8501

WORKDIR /workspace

# app.* を import できるようにする
#（streamlit run はスクリプトのあるディレクトリをsys.pathに入れるだけでCWDは入れないため、PYTHONPATH の指定が必須）
ENV PYTHONPATH=/workspace \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies and SQLite CLI
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# non-root user作成
ARG USERNAME=developer
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} ${USERNAME} \
    && useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME}

# 名前付きボリュームのマウント先をイメージ側に作り、所有者を非rootにする（root所有だと非rootユーザーがSQLiteを書けないため）
RUN mkdir -p /db /derived /workspace \
    && chown -R ${UID}:${GID} /db /derived /workspace
COPY --chown=${UID}:${GID} ./docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

USER ${USERNAME}

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
# ENTRYPOINT ではなく CMD に置くことで、
#   docker compose run --rm webapp python -m app.json_conversion.to_nusc_db ...
# のようにコマンドを差し替えてインポート CLI を実行できる。
CMD ["streamlit", "run", "app/streamlit/main.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
