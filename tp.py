import pika

try:
    connection = pika.BlockingConnection(pika.ConnectionParameters(
        host='localhost',
        port=5672,
        credentials=pika.PlainCredentials('guest', 'guest')
    ))
    print("Connection successful")
    connection.close()
except Exception as e:
    print(f"Failed to connect: {e}")
