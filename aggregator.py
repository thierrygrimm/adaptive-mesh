import time
import json
import glob

class MetricAggregator:
    def __init__(self, config):
        self.config = config
        self.data = []

    def add(self, packet, ack):
        self.data.append((packet, ack))

    def aggregate(self):
        # Aggregate over last 1 min
        now = time.time()
        window = [d for d in self.data if now - d[0]['timestamp'] < 60]
        # Compute averages and stats
        return self.evaluate_performance(window)

    def aggregate_metrics(self):
        """
        Aggregate all sent_packets_*.json files and their ACK metrics.
        Returns a dict with average RTT, RSSI, SNR, delivery rate, etc.
        """
        files = glob.glob('sent_packets_*.json')
        all_packets = []
        for file in files:
            try:
                with open(file, 'r') as f:
                    all_packets.extend(json.load(f))
            except Exception as e:
                print(f"[ERROR] Failed to load {file}: {e}")
        if not all_packets:
            return {}
        rtts = [pkt['rtt_ms'] for pkt in all_packets if 'rtt_ms' in pkt and pkt['rtt_ms'] is not None]
        ack_rssis = [float(pkt['ack_rssi']) for pkt in all_packets if 'ack_rssi' in pkt and pkt['ack_rssi'] is not None]
        ack_snrs = [float(pkt['ack_snr']) for pkt in all_packets if 'ack_snr' in pkt and pkt['ack_snr'] is not None]
        pkt_rssis = [float(pkt['packet_rssi']) for pkt in all_packets if 'packet_rssi' in pkt and pkt['packet_rssi'] is not None]
        pkt_snrs = [float(pkt['packet_snr']) for pkt in all_packets if 'packet_snr' in pkt and pkt['packet_snr'] is not None]
        delivered = sum(1 for pkt in all_packets if 'rtt_ms' in pkt and pkt['rtt_ms'] is not None)
        total = len(all_packets)
        metrics = {
            'avg_rtt_ms': sum(rtts)/len(rtts) if rtts else None,
            'avg_ack_rssi': sum(ack_rssis)/len(ack_rssis) if ack_rssis else None,
            'avg_ack_snr': sum(ack_snrs)/len(ack_snrs) if ack_snrs else None,
            'avg_packet_rssi': sum(pkt_rssis)/len(pkt_rssis) if pkt_rssis else None,
            'avg_packet_snr': sum(pkt_snrs)/len(pkt_snrs) if pkt_snrs else None,
            'delivery_rate': delivered/total if total else 0,
            'total_packets': total,
            'delivered_packets': delivered
        }
        return metrics

    def get_policy_input(self):
        """
        Return a summary of metrics for RL input (e.g., as a dict).
        """
        return self.aggregate_metrics()

    def evaluate_performance(self, window):
        # Compute metrics for RL input
        return {}
