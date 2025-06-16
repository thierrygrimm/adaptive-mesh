import time
import json
import glob
import os

class MetricAggregator:
    """Aggregates network performance metrics from packet and ACK data."""
    
    def __init__(self, config):
        """Initialize the metric aggregator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data = []

    def add(self, packet, ack):
        """Add packet and ACK data to the aggregator.
        
        Args:
            packet: Dictionary containing packet data
            ack: Dictionary containing ACK data
        """
        self.data.append((packet, ack))

    def aggregate(self):
        """Aggregate metrics over the last 60 seconds.
        
        Returns:
            Dictionary containing aggregated performance metrics
        """
        # Aggregate over last 1 min
        now = time.time()
        window = [d for d in self.data if now - d[0]['timestamp'] < 60]
        # Compute averages and stats
        return self.evaluate_performance(window)

    def aggregate_from_memory(self, sent_packets, ack_metrics):
        """Aggregate metrics from in-memory sent_packets and ack_metrics dicts.
        
        Args:
            sent_packets: Dictionary mapping msg_id to packet data
            ack_metrics: Dictionary mapping msg_id to ACK data
            
        Returns:
            Dictionary containing average RTT, RSSI, SNR, delivery rate, etc.
        """
        all_packets = []
        for msg_id, pkt in sent_packets.items():
            pkt_copy = pkt.copy()
            ack = ack_metrics.get(msg_id)
            if ack:
                pkt_copy.update(ack)
            all_packets.append(pkt_copy)
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
        print(f"[AGGREGATE] In-memory: {delivered}/{total} packets delivered ({metrics['delivery_rate']:.1%})")
        return metrics

    def get_policy_input(self):
        """Return a summary of metrics for RL input.
        
        Returns:
            Dictionary containing aggregated metrics for reinforcement learning
        """
        return self.aggregate()

    def evaluate_performance(self, window):
        """Compute metrics for RL input.
        
        Args:
            window: List of packet/ACK data tuples
            
        Returns:
            Dictionary containing performance metrics
        """
        # Compute metrics for RL input
        return {}
