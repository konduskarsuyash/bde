# debug_env.py
import os
import subprocess
import sys

def run_command(command):
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"

def check_environment():
    checks = {
        "Java Version": "java -version",
        "Java Home": "echo $JAVA_HOME",
        "Java Path": "which java",
        "Java Home Contents": "ls -l $JAVA_HOME/bin",
        "Spark Home": "echo $SPARK_HOME",
        "Spark Contents": "ls -l $SPARK_HOME/bin",
        "Python Version": "python --version",
        "Python Path": "which python",
        "System Path": "echo $PATH",
        "System Memory": "free -h",
        "Directory Structure": "ls -R /usr/lib/jvm/"
    }
    
    results = {}
    for check, command in checks.items():
        print(f"\n=== {check} ===")
        output = run_command(command)
        print(output)
        results[check] = output
    
    # Additional specific checks
    spark_classpath = os.environ.get('SPARK_CLASSPATH', 'Not Set')
    print("\n=== Spark Classpath ===")
    print(spark_classpath)
    
    # Check if critical directories exist
    critical_paths = [
        '/usr/lib/jvm/java-11-openjdk-amd64',
        '/opt/spark',
        '/opt/spark/conf',
        '/opt/spark/jars'
    ]
    
    print("\n=== Critical Path Checks ===")
    for path in critical_paths:
        exists = os.path.exists(path)
        print(f"{path}: {'Exists' if exists else 'Missing'}")

if __name__ == "__main__":
    check_environment()