# app/Dockerfile
# Based on https://docs.streamlit.io/deploy/tutorials/docker

# First, build this container with the 4 commands below in your terminal:
# $ export DOCKER_BUILDKIT=1
# $ eval `ssh-agent`
# $ ssh-add ~/.ssh/<key here>
# $ docker build -t "streamlit" --ssh default .

# Then, run the container with this command:
# $ docker run -p 8501:8501 streamlit

FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Tell docker what hosts it can connect to and hash them so it isn't in plaintext
RUN mkdir -p -m 0600 ~/.ssh && \
    ssh-keyscan -H github.com >> ~/.ssh/known_hosts

# Mount the ssh-agent so it can install from private repos 
#   Note: access to ao_core requires a private beta license; request yours via https://calendly.com/aee/aolabs or https://discord.com/invite/nHuJc4Y4n7
RUN --mount=type=ssh \
    pip install git+ssh://git@github.com/aolabsai/ao_core.git \
                git+git://github.com/aolabsai/ao_arch.git 

COPY . /app

# Install standard requirements
RUN pip install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
