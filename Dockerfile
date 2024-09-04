# app/Dockerfile
# Based on https://docs.streamlit.io/deploy/tutorials/docker

# built with below 4 line command
# export DOCKER_BUILDKIT=1
#eval `ssh-agent`
#ssh-add ~/.ssh/<key here>
#docker build -t "streamlit" --ssh default .

# Run with below command
#docker run -p 8501:8501 streamlit

FROM python:3.12-slim

WORKDIR /app


RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

#Tell docker what hosts it can connect to and hash them so it isn't in plaintext
RUN mkdir -p -m 0600 ~/.ssh && \
    ssh-keyscan -H github.com >> ~/.ssh/known_hosts

#Mount the ssh agent so it can install from a private repo 
RUN --mount=type=ssh \
    pip install git+ssh://git@github.com/aolabsai/ao_core.git \
                git+ssh://git@github.com/aolabsai/ao_arch.git 

COPY . /app

#install standard requirements 
RUN pip install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
