# Use Python 3.12 slim base image
FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Copy and install Python dependencies
COPY deps.txt .
RUN pip install --no-cache-dir -r deps.txt

# Copy the rest of your code
COPY . .

# Expose the default Streamlit port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "webapp.py", "--server.address=0.0.0.0"]
