# Use the official Python image from the Docker Hub
FROM python:3.8

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the content of the local src directory to the /app directory
COPY . .

# Specify the command to run on container start
CMD ["streamlit", "run", "your_app_file.py"]
