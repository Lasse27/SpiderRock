import socket

HOST = "0.0.0.0"
PORT = 514
BUFFER_SIZE = 4096

if __name__ == "__main__":
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((HOST, PORT))
    print(f"UDP server listening on {HOST}:{PORT}")

    while True:
        data, address = sock.recvfrom(BUFFER_SIZE)

        message = data.decode("utf-8", errors="replace")
        print(f"[{address[0]}:{address[1]}] {message}")

