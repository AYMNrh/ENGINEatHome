FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git patch diffutils findutils grep sed gawk less make && rm -rf /var/lib/apt/lists/* && pip install --no-cache-dir pytest freezegun && useradd -m -u 1000 agent
USER agent
WORKDIR /work
