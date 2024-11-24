FROM python:3.11-slim-buster

# Install dependencies and OpenJDK 11
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-11-jdk \
    libgomp1 \
    wget \
    locales \
    ca-certificates-java \
    procps && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set locale
RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && locale-gen
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# Set JAVA_HOME and Spark environment variables
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH
ENV SPARK_HOME=/opt/spark
ENV PATH=$SPARK_HOME/bin:$PATH
ENV PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.9.7-src.zip:$PYTHONPATH
ENV PYSPARK_PYTHON=/usr/local/bin/python
ENV PYSPARK_DRIVER_PYTHON=/usr/local/bin/python
ENV PYSPARK_SUBMIT_ARGS="--master local[3] --driver-java-options=-Xms1024M --driver-java-options=-Xmx4096M pyspark-shell"

# Verify Java installation
RUN java -version

# Download and install Spark 3.5.3
RUN wget https://archive.apache.org/dist/spark/spark-3.5.3/spark-3.5.3-bin-hadoop3.tgz && \
    tar -xzf spark-3.5.3-bin-hadoop3.tgz && \
    mv spark-3.5.3-bin-hadoop3 /opt/spark && \
    rm spark-3.5.3-bin-hadoop3.tgz

# Create Spark temporary directory
RUN mkdir -p /tmp/spark-temp && chmod -R 777 /tmp/spark-temp

# Add a basic log4j configuration
RUN echo "log4j.rootCategory=INFO, console" > $SPARK_HOME/conf/log4j.properties

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
