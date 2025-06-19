import time
import random
import utils

class PacketGenerator:
    """Generates packets for transmission over the mesh network."""
    
    def __init__(self, config, serial_interface, log_file='sent_packets.json'):
        """Initialize the packet generator.
        
        Args:
            config: Configuration dictionary
            serial_interface: Serial interface object
            log_file: Path to log file for sent packets (default: 'sent_packets.json')
        """
        self.config = config
        self.serial_interface = serial_interface
        self.packet_id = 0
        self.log_file = log_file

    def create_packet(self, payload_size=16):
        """Create a packet with random payload.
        
        Args:
            payload_size: Size of the payload in bytes (default: 16)
            
        Returns:
            Dictionary containing packet data with id, payload, and timestamp
        """
        payload = bytes(random.getrandbits(8) for _ in range(payload_size))
        packet = {
            'id': self.packet_id,
            'payload': payload,
            'timestamp': utils.current_time()
        }
        self.packet_id += 1
        return packet

def send_packet(packet, serial_interface, log_file='sent_packets.json'):
    """Send a single packet using the serial interface.
    
    Args:
        packet: Dictionary containing packet data
        serial_interface: Serial interface object
        log_file: Path to log file (default: 'sent_packets.json')
    """
    serial_interface.sendText(packet['payload'])
    utils.log_send(packet, log_file)

def send_broadcast_packets(serial_interface, duration=60, interval=0.1, log_file=None, idle_time=10, start_msg_id=0):
    """Send broadcast packets for a given duration using sendData.
    
    Args:
        serial_interface: Serial interface object
        duration: Duration to send packets in seconds (default: 60)
        interval: Interval between packets in seconds (default: 0.1)
        log_file: Path to log file (optional, no longer used)
        idle_time: Time to wait after sending to collect ACKs (default: 10)
        start_msg_id: Starting message ID (default: 0)
        
    Returns:
        List of sent packet dictionaries for global indexing
    """
    import unicodedata
    import os
    # log_file is no longer required or used
    start_time = time.time()
    msg_id = start_msg_id
    sent_packets = []
    # Try to get LoRa parameters from the device
    lora_params = {}
    try:
        lora = serial_interface.localNode.localConfig.lora
        lora_params['spreading_factor'] = getattr(lora, 'spread_factor', None)
        lora_params['bandwidth'] = getattr(lora, 'bandwidth', None)
    except Exception:
        # Fallback: try config if available
        try:
            import config as config_module
            cfg = getattr(serial_interface, 'config', None)
            if cfg is None:
                cfg = getattr(config_module, 'DEFAULT_CONFIG', {})
            lora_params['spreading_factor'] = cfg.get('lora_params', {}).get('spreading_factor')
            lora_params['bandwidth'] = cfg.get('lora_params', {}).get('bandwidth')
        except Exception:
            lora_params['spreading_factor'] = None
            lora_params['bandwidth'] = None
    while time.time() - start_time < duration:
        message = f"MSG_{msg_id}_{int(time.time())}"
        log_entry = {
            'id': msg_id,
            'payload': message,
            'timestamp': utils.current_time(),
            'spreading_factor': lora_params['spreading_factor'],
            'bandwidth': lora_params['bandwidth']
        }
        # Update in-memory SENT_PACKETS dict before sending
        try:
            import main
            main.SENT_PACKETS[str(log_entry['id'])] = log_entry
        except Exception as e:
            import logging
            logging.warning(f"Could not update SENT_PACKETS in main: {e}")
        print(f"Sent:   MSG_{log_entry['id']} (payload: {log_entry['payload']})")
        serial_interface.sendData(message.encode('utf-8'))  # Broadcast: no destinationId
        sent_packets.append(log_entry)
        msg_id += 1
        time.sleep(interval)
    # Idle wait to collect ACKs/messages
    print(f"[SENDER] Idle for {idle_time} seconds to collect incoming messages...")
    time.sleep(idle_time)
    return sent_packets
