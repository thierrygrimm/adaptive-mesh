import time
import json
import csv
import os

def log_send(packet, log_path='sent_packets.json'):
    print(f"[SEND] {packet}")
    # Append packet to JSON log file
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []
    data.append(packet)
    with open(log_path, 'w') as f:
        json.dump(data, f, indent=2)

def update_json_metrics(file, msg_id, ack_metrics, rtt_ms):
    try:
        with open(file, 'r') as f:
            data = json.load(f)
        for pkt in data:
            if str(pkt.get('id')) == str(msg_id):
                pkt['ack_rssi'] = ack_metrics['ack_rssi']
                pkt['ack_snr'] = ack_metrics['ack_snr']
                pkt['packet_rssi'] = ack_metrics['packet_rssi']
                pkt['packet_snr'] = ack_metrics['packet_snr']
                pkt['rtt_ms'] = rtt_ms
        with open(file, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[ERROR] Failed to update JSON metrics for msg_id={msg_id}: {e}")

def log_receive(packet, ack):
    print(f"[RECV] {packet} | ACK: {ack}")

def current_time():
    return time.time()

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)

def save_csv(data, path):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)
