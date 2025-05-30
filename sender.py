import time
import random
import utils

class PacketGenerator:
    def __init__(self, config, serial_interface, log_file='sent_packets.json'):
        self.config = config
        self.serial_interface = serial_interface
        self.packet_id = 0
        self.log_file = log_file

    def create_packet(self, payload_size=16):
        payload = bytes(random.getrandbits(8) for _ in range(payload_size))
        packet = {
            'id': self.packet_id,
            'payload': payload,
            'timestamp': utils.current_time()
        }
        self.packet_id += 1
        return packet

def send_packet(packet, serial_interface, log_file='sent_packets.json'):
    serial_interface.sendText(packet['payload'])
    utils.log_send(packet, log_file)

def send_broadcast_packets(serial_interface, duration=60, interval=0.1, log_file='sent_packets.json', idle_time=10):
    """
    Send broadcast packets for a given duration (seconds) using sendData.
    Each packet has a unique message ID and is encoded as UTF-8.
    After sending, wait idle for idle_time seconds to collect incoming messages/ACKs.
    This function is repeatable and can be called in a loop.
    """
    start_time = time.time()
    msg_id = 0
    while time.time() - start_time < duration:
        message = f"MSG_{msg_id}_{int(time.time())}"
        serial_interface.sendData(message.encode('utf-8'))  # Broadcast: no destinationId
        utils.log_send({'id': msg_id, 'payload': message, 'timestamp': utils.current_time()}, log_file)
        msg_id += 1
        time.sleep(interval)
        
    # Idle wait to collect ACKs/messages
    print(f"[SENDER] Idle for {idle_time} seconds to collect incoming messages...")
    time.sleep(idle_time)
