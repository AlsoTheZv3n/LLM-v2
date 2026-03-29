FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3-pip \
    git curl tmux htop nvtop \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python

# Python deps
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124

# Copy project
COPY training/ training/
COPY agents/ agents/
COPY infra/ infra/
COPY CLAUDE.md .

# Data volume
VOLUME /data/tokenized
VOLUME /app/runs

EXPOSE 8080

ENTRYPOINT ["python"]
CMD ["training/train.py", "--config", "1b", "--no-plot", "--wandb"]
