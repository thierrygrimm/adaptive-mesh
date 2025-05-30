import utils
from pubsub import pub
import json
import glob
import os

class PacketReceiver:
    def __init__(self, config, serial_interface):
        self.config = config
        self.serial_interface = serial_interface

    def listen(self):
        while True:
            packet = self.serial_interface.receive()
            if packet:
                self.handle_packet(packet)

    def handle_packet(self, packet):
        ack = parse_ack(packet)
        utils.log_receive(packet, ack)
        # Application-layer ACK for MSG_ packets
        payload = packet.get('payload')
        if payload:
            try:
                # Try to decode as utf-8 if bytes
                if isinstance(payload, bytes):
                    payload_str = payload.decode('utf-8', errors='ignore')
                else:
                    payload_str = str(payload)
                if payload_str.startswith('MSG_'):
                    # Extract message ID (format: MSG_<id>_<timestamp>)
                    parts = payload_str.split('_')
                    if len(parts) >= 3:
                        msg_id = parts[1]
                        # Prepare ACK payload
                        rssi = packet.get('rxRssi')
                        snr = packet.get('rxSnr')
                        ack_payload = f"ACK_{msg_id}_RSSI_{rssi}_SNR_{snr}"
                        # Send ACK directly to sender (not broadcast)
                        sender_id = packet.get('fromId')
                        if sender_id:
                            self.serial_interface.sendData(ack_payload.encode('utf-8'), destinationId=sender_id)
                        else:
                            print(f"[ERROR] No sender_id found for MSG_ packet: {payload_str}")
                elif payload_str.startswith('ACK_'):
                    # Parse ACK payload: ACK_<id>_RSSI_<rssi>_SNR_<snr>
                    parts = payload_str.split('_')
                    if len(parts) >= 6:
                        msg_id = parts[1]
                        ack_rssi = parts[3]
                        ack_snr = parts[5]
                        # Also get RSSI/SNR from the received ACK packet
                        packet_rssi = packet.get('rxRssi')
                        packet_snr = packet.get('rxSnr')
                        # Calculate RTT
                        send_time = self.get_send_time_for_msg_id(msg_id, return_file=True)
                        rtt_ms = None
                        ack_metrics = {
                            'ack_rssi': ack_rssi,
                            'ack_snr': ack_snr,
                            'packet_rssi': packet_rssi,
                            'packet_snr': packet_snr
                        }
                        if send_time and send_time[0] is not None:
                            rtt_ms = (utils.current_time() - send_time[0]) * 1000
                            print(f"[ACK RECEIVED] msg_id={msg_id} | RTT={rtt_ms:.2f} ms | ACK RSSI={ack_rssi}, ACK SNR={ack_snr} | Packet RSSI={packet_rssi}, Packet SNR={packet_snr}")
                        else:
                            print(f"[ACK RECEIVED] msg_id={msg_id} | RTT=UNKNOWN | ACK RSSI={ack_rssi}, ACK SNR={ack_snr} | Packet RSSI={packet_rssi}, Packet SNR={packet_snr}")
                        # Update JSON file with metrics
                        if send_time and send_time[1]:
                            utils.update_json_metrics(send_time[1], msg_id, ack_metrics, rtt_ms)
            except Exception as e:
                print(f"[ERROR] Failed to process MSG_/ACK_ packet: {e}")

    def get_send_time_for_msg_id(self, msg_id, return_file=False):
        # Find the latest sent_packets_*.json file
        files = sorted(glob.glob('sent_packets_*.json'), reverse=True)
        for file in files:
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    for pkt in data:
                        # Match by id (as string or int)
                        if str(pkt.get('id')) == str(msg_id):
                            if return_file:
                                return pkt.get('timestamp'), file
                            return pkt.get('timestamp')
            except Exception as e:
                continue
        if return_file:
            return None, None
        return None


def parse_ack(packet):
    # Extract metrics from ACK or packet
    ack = {
        'rssi': packet.get('rxRssi'),
        'snr': packet.get('rxSnr'),
        'ack': packet.get('ack', False)
    }
    return ack
