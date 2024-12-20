FROM python:3.11-slim-buster

# Set the working directory to /mvsp
WORKDIR /mvsp

# Copy the pyproject.toml and requirements.txt files to the container
COPY . .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Start the web server
CMD ["python", "-m", "mvsp.main"]