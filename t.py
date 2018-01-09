import bluetooth as b

d = b.discover_devices(duration=1)
print d

client_socket=b.BluetoothSocket( b.RFCOMM )

# print client_socket.getpeername()

client_socket.connect(("00:16:53:17:EF:0A", 1))

client_socket.send("Hello World")

# print "Finished"

# print(dir(client_socket))

print client_socket.getpeername()[0]


client_socket.close()

