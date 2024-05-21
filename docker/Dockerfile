# Use a base image with Python
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Copy the application code
COPY app /app

# Command to run the script
# CMD ["python3", "app/main.py"]

#Comand to let the prompt opened
ENTRYPOINT ["tail", "-f", "/dev/null"]
