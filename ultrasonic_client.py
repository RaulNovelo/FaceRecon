
from socket import *
import time
import sys
import os

# create a socket and bind socket to the host
client_socket = socket(AF_INET, SOCK_STREAM)
client_socket.connect((sys.argv[1], 8002))

try:
    while True:
        try:
            sensor_data = client_socket.recv(1024)
            print(sensor_data)
        except:
            pass
        
finally:
    client_socket.close()
