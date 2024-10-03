# ao_app/Dockerfile

# First, build this container by running the 2 commands below in a Git Bash terminal:
# $ export DOCKER_BUILDKIT=1
# $ docker build --secret id=env,src=.env -t "ao_app" .

# Then, run the container with this command:
# $ docker run -p 8501:8501 "ao_app"

# You can then access your app at: http://localhost:8501/


FROM python:3.12-slim

# Create a directory for the app in the container
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install git and other necessary packages
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the app code including the requirements file
COPY . /app
# COPY misc/index.html /usr/local/lib/python3.12/site-packages/streamlit/static/index.html

# Install dependencies from the requirements file
RUN pip install -r requirements.txt

RUN python add_meta_tags.py

# Install AO modules, ao_core and ao_arch
#    Notes: - ao_core is a private repo; say hi for access: https://calendly.com/aee/aolabs or https://discord.com/invite/nHuJc4Y4n7
#           - already have access? generate your Personal Access Token from github here: https://github.com/settings/tokens?type=beta 
RUN --mount=type=secret,id=env,target=/app/.env \
    export $(grep -v '^#' .env | xargs) && \
    pip install git+https://${ao_github_PAT}@github.com/aolabsai/ao_core.git
RUN pip install git+https://github.com/aolabsai/ao_arch.git

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501"]
