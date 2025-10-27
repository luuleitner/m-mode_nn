# Operating System - Use Ubuntu 24.04 LTS as base image
FROM ubuntu:24.04

# Metadata
LABEL maintainer="chu.leitner@gmail.com"
LABEL description="CPU: ML environment based on Ubuntu 24.04"

# Set non-interactive mode (skip interactive prompts during install)
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    ca-certificates \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    libjpeg-dev \
    libpng-dev \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Create a working directory
WORKDIR /lab

# Copy requirements.txt if you have one
COPY requirements.txt .

# Alternatively, uncomment the below line to install from a requirements file
RUN pip install -r requirements.txt

# Set default command
CMD [ "python3" ]