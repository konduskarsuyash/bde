FROM python:3.11-slim-buster

# Install dependencies and OpenJDK 11
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    openjdk-11-jdk \
    libgomp1 \
    wget \
    ca-certificates-java && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME and environment variables
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH
ENV SPARK_HOME=/opt/spark
ENV PATH=$SPARK_HOME/bin:/usr/local/bin:$PATH
ENV PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.9.7-src.zip:$PYTHONPATH
ENV PYSPARK_PYTHON=/usr/local/bin/python
ENV PYSPARK_DRIVER_PYTHON=/usr/local/bin/python
ENV PYSPARK_SUBMIT_ARGS="--driver-java-options=-Xms1024M --driver-java-options=-Xmx4096M --driver-java-options=-Dlog4j.logLevel=info pyspark-shell"

# Download and install Spark 3.5.3
RUN wget https://archive.apache.org/dist/spark/spark-3.5.3/spark-3.5.3-bin-hadoop3.tgz \
    && tar -xzf spark-3.5.3-bin-hadoop3.tgz \
    && mv spark-3.5.3-bin-hadoop3 /opt/spark \
    && rm spark-3.5.3-bin-hadoop3.tgz

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit's default ports
EXPOSE 8501
EXPOSE 8502

# Default command
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
