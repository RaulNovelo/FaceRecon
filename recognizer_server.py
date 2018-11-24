import os
import time
import socket
import sys
from threading import Thread

class Consumer(Thread):
    def __init__(self, host, port):
        self.server_socket = socket.socket()
        self.server_socket.setsockopt(
            socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((host, port))
        self.server_socket.listen(0)
        self.connection, self.client_address = self.server_socket.accept()
        self.host_name = socket.gethostname()
        self.host_ip = socket.gethostbyname(self.host_name)
        self.run()

    def run(self):
        try:
            print("Host: ", self.host_name + ' ' + self.host_ip)
            print("Connection from: ", self.client_address)

            while True:
                sensor_data = self.connection.recv(1024)
                print(sensor_data)

        finally:
            self.connection.close()
            self.server_socket.close()

if __name__ == "__main__":
    Consumer(sys.argv[1], 8002).start()
