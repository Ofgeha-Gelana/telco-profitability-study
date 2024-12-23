# Use the official Python base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project to the container
COPY . .

# Expose the port your application runs on (if applicable)
# EXPOSE 8000 

# Set the default command to run your application (customize as needed)
CMD ["python", "-m", "src"]
