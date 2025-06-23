# =====================
# Imports
# =====================
import argparse
import random
import time
import threading
import base64
import json
import yaml
import os
import logging
import unicodedata
import serial
from typing import Optional, Dict, Any
from google.protobuf.json_format import MessageToDict
from nacl.public import PrivateKey, PublicKey, Box
from nacl.exceptions import CryptoError
from pubsub import pub

import config
import sender
import mesh
import aggregator
import parameter_manager
import utils
from receiver import PacketReceiver

import meshtastic
import meshtastic.serial_interface

try:
    import colorlog
except ImportError:
    colorlog = None

# --- Encryption/Decryption code is now disabled as per user request ---
# from Crypto.Cipher import AES
# CHANNEL_KEY_B64 = None
# def extract_channel_key_from_config(cfg):
#     ...
# def decrypt_aes_ctr_channel_packet(encrypted_bytes):
#     ...
# def try_decrypt_ack(packet):
#     ...

# Setup logging configuration at the top of the file, after imports
if colorlog:
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s[%(levelname)s]%(reset)s %(message)s',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        }
    ))
    logging.root.handlers = [handler]
    logging.root.setLevel(logging.INFO)  # Or logging.WARNING for even less output
else:
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

logging.getLogger("meshtastic").setLevel(logging.WARNING)
logging.getLogger("serial").setLevel(logging.WARNING)
# Add other noisy libraries as needed

def print_system_message(message):
    """Print a system message in light blue.
    
    Args:
        message: Text to display in light blue color
    """
    print(f"\033[94m{message}\033[0m")

def print_cycle_start(message):
    """Print a cycle start message in dark blue.
    
    Args:
        message: Text to display in dark blue color
    """
    print(f"\033[34m{message}\033[0m")

def print_info_box(title, content, width=60):
    """Print information in a nice ASCII box.
    
    Args:
        title: Box title text
        content: Content text to display in box
        width: Box width in characters (default: 60)
    """
    print_system_message("╔" + "═" * (width - 2) + "╗")
    print_system_message("║" + title.center(width - 2) + "║")
    print_system_message("╠" + "═" * (width - 2) + "╣")
    for line in content.split('\n'):
        if line.strip():
            print_system_message("║ " + line.ljust(width - 4) + " ║")
    print_system_message("╚" + "═" * (width - 2) + "╝")

def print_section_header(title, width=60):
    """Print a section header with ASCII art.
    
    Args:
        title: Section title text
        width: Header width in characters (default: 60)
    """
    print_system_message("")
    print_system_message("┌" + "─" * (width - 2) + "┐")
    print_system_message("│ " + title.ljust(width - 4) + " │")
    print_system_message("└" + "─" * (width - 2) + "┘")
    print_system_message("")

def print_green_section_header(title, width=60):
    """Print a section header in green.
    
    Args:
        title: Section title text
        width: Header width in characters (default: 60)
    """
    print(f"\033[92m")  # Green color
    print("┌" + "─" * (width - 2) + "┐")
    print("│ " + title.ljust(width - 4) + " │")
    print("└" + "─" * (width - 2) + "┘")
    print(f"\033[0m")  # Reset color

def print_yellow_section_header(title, width=60):
    """Print a section header in yellow.
    
    Args:
        title: Section title text
        width: Header width in characters (default: 60)
    """
    print(f"\033[93m")  # Yellow color
    print("┌" + "─" * (width - 2) + "┐")
    print("│ " + title.ljust(width - 4) + " │")
    print("└" + "─" * (width - 2) + "┘", end="")  # Remove newline
    print(f"\033[0m")  # Reset color

def print_cycle_header_box(cycle_num, total_cycles, sf, bw, cr):
    """Print cycle header information in a red box.
    
    Args:
        cycle_num: Current cycle number
        total_cycles: Total number of cycles
        sf: Spreading factor value
        bw: Bandwidth value
        cr: Coding rate value
    """
    print(f"\033[91m")  # Red color
    
    # Create the content line
    content = f"LoRa Params: SF={sf}, BW={bw}, CR={cr}"
    
    # Calculate the width needed for the box
    title = f"Cycle {cycle_num}/{total_cycles}"
    width = max(len(title) + 6, len(content) + 4)  # Add padding
    
    # Print the top border
    print("┌" + "─" * (width - 2) + "┐")
    
    # Print the title line
    print("│ " + title.ljust(width - 4) + " │")
    
    # Print the content line
    print("│ " + content.ljust(width - 4) + " │")
    
    # Print the bottom border
    print("└" + "─" * (width - 2) + "┘")
    
    print(f"\033[0m")  # Reset color

def print_cycle_box(cycle_num, rtt_str, delivery_str, success_metric):
    """Print cycle results in a white box.
    
    Args:
        cycle_num: Current cycle number
        rtt_str: RTT string representation
        delivery_str: Delivery rate string representation
        success_metric: Success metric value
    """
    print(f"\033[97m")  # White color
    
    # Create the content line
    content = f"RTT={rtt_str} ms, Delivery={delivery_str}, Success={success_metric:.3f}"
    
    # Calculate the width needed for the box
    title = f"Cycle {cycle_num}"
    width = max(len(title) + 6, len(content) + 4)  # Add padding
    
    # Print the top border
    print("┌" + "─" * (width - 2) + "┐")
    
    # Print the title line
    print("│ " + title.ljust(width - 4) + " │")
    
    # Print the content line
    print("│ " + content.ljust(width - 4) + " │")
    
    # Print the bottom border
    print("└" + "─" * (width - 2) + "┘")
    
    print(f"\033[0m")  # Reset color

# =====================
# Message Handling
# =====================
NODE_ID_TO_PUBKEY = {}

def get_all_node_public_keys(interface) -> Dict[str, str]:
    """Get a mapping from nodeId to publicKey for all nodes in the mesh network.
    
    Args:
        interface: Meshtastic interface object
        
    Returns:
        Dictionary mapping node IDs to public keys
    """
    try:
        mapping = {}
        if interface.nodes:
            for node_id, node_info in interface.nodes.items():
                user_info = node_info.get('user', {})
                pubkey = user_info.get('publicKey')
                if pubkey:
                    mapping[node_id] = pubkey
        return mapping
    except Exception as e:
        logging.error(f"[ERROR] Could not get node public keys: {e}")
        return {}

def get_specific_node_public_key(node_id: str, interface) -> Optional[str]:
    """Get public key for a specific node.
    
    Args:
        node_id: Node identifier
        interface: Meshtastic interface object
        
    Returns:
        Public key string or None if not found
    """
    try:
        public_key = None
        if interface.nodes and node_id in interface.nodes:
            user_info = interface.nodes[node_id].get('user', {})
            public_key = user_info.get('publicKey')
        return public_key
    except Exception as e:
        logging.error(f"[ERROR] Could not get public key for node {node_id}: {e}")
        return None

# In main(), do NOT call extract_channel_key_from_config(cfg)
# Also, comment out any debug output or logic related to decryption.

def try_decrypt_ack(packet):
    """Attempt to decrypt ACK packet (currently disabled).
    
    Args:
        packet: Packet to decrypt
        
    Returns:
        None (decryption disabled)
    """
    # Encryption/decryption is now ignored. All packets are treated as plaintext.
    return None

def handle_incoming_message(packet, interface=None):
    msg_type, display_text = PacketReceiver.filter_and_format_packet(packet)
    special_info = PacketReceiver.handle_special_payload(packet, handle_incoming_message.serial_interface)
    def is_printable(s):
        try:
            if isinstance(s, bytes):
                s = s.decode('utf-8')
            s.encode('utf-8')
            # Check for non-printable characters
            return all(32 <= ord(c) < 127 or c in '\n\r\t' for c in s)
        except Exception:
            return False
    
    # Check if this is a binary prediction message first
    decoded = packet.get('decoded', {})
    payload = decoded.get('payload') if decoded else packet.get('payload')
    if payload and isinstance(payload, bytes) and payload.startswith(b'PRED'):
        # Handle binary prediction messages - don't log as text
        logging.debug(f"[RECEIVED PREDICTION] Binary prediction message: {len(payload)} bytes from {packet.get('fromId', 'unknown')}")
        return  # Skip text logging for binary prediction messages
    
    if msg_type == 'text':
        if is_printable(display_text):
            logging.info(f"[RECEIVED TEXT] {display_text}")
    elif msg_type == 'payload':
        if is_printable(display_text):
            logging.info(f"[RECEIVED PAYLOAD] {display_text}")
        else:
            logging.debug(f"[RECEIVED BINARY] Non-printable payload: {len(display_text) if hasattr(display_text, '__len__') else 'unknown'} bytes")
    elif msg_type == 'ack':
        logging.info(f"[RECEIVED ACK] {display_text}")
        if special_info:
            logging.info(f"[ACK METRICS] {special_info}")
        if 'encrypted' in packet:
            result = try_decrypt_ack(packet)
            if result is None:
                logging.warning("[DECRYPT] Could not decrypt ACK.")
    else:
        # Concise unknown/encrypted packet debug
        logging.debug(f"[RECEIVED UNKNOWN ENCRYPTED PACKET] fromId: {packet.get('fromId')} toId: {packet.get('toId')} id: {packet.get('id')}")
        if 'encrypted' in packet:
            result = try_decrypt_ack(packet)
            if result is None:
                logging.warning("[DECRYPT] Could not decrypt unknown packet.")
    if special_info and msg_type not in ('ack'):
        logging.debug(f"[DEBUG] {special_info}")

handle_incoming_message.serial_interface = None

# =====================
# Experiment Loop
# =====================
# --- Global message id counter ---
GLOBAL_MSG_ID = 0

# Global index for msg_id to log file
MSG_ID_TO_LOGFILE = {}

# Remove MSG_ID_TO_LOGFILE and all per-packet file logic
# Add a global in-memory sent_packets dict
SENT_PACKETS = {}

import unicodedata

def get_current_lora_params(mesh_iface):
    """Get current LoRa parameters from the device.
    
    Args:
        mesh_iface: Meshtastic mesh interface object
        
    Returns:
        Dictionary of current LoRa parameters
    """
    node = mesh_iface.get_local_node()
    if node is None:
        return {}
    
    lora = node.localConfig.lora
    params = {}
    if hasattr(lora, 'spread_factor'):
        params['spreading_factor'] = lora.spread_factor
    if hasattr(lora, 'bandwidth'):
        params['bandwidth'] = lora.bandwidth
    if hasattr(lora, 'coding_rate'):
        params['coding_rate'] = lora.coding_rate
    return params

def aggregate_cycle_metrics(node_dir, cycle_num, current_params, aggregator_mod):
    """Aggregate metrics for a specific cycle and parameter configuration.
    
    Args:
        node_dir: Directory where node-specific data is stored
        cycle_num: Current cycle number
        current_params: Current LoRa parameter configuration
        aggregator_mod: MetricAggregator module
        
    Returns:
        Aggregated metrics dictionary or None if no metrics found
    """
    logging.info(f"[AGGREGATE] Processing cycle {cycle_num} with params: {current_params}")
    
    # Find the sent_packets file for this cycle
    sent_file = os.path.join(node_dir, f'sent_packets{cycle_num}.json')
    if not os.path.exists(sent_file):
        logging.warning(f"[WARN] No sent packets file found for cycle {cycle_num}: {sent_file}")
        return None
    
    # Aggregate metrics for this specific file
    metrics = aggregator_mod.aggregate_specific_file(sent_file)
    if metrics:
        # Add parameter configuration and cycle info
        metrics['cycle'] = cycle_num
        metrics['parameters'] = current_params
        metrics['timestamp'] = time.time()
        return metrics
    return None

def save_aggregated_results(all_metrics, node_id, output_dir):
    """Save aggregated results to a JSON file with parameter stratification.
    
    Args:
        all_metrics: List of aggregated metrics dictionaries
        node_id: Node identifier
        output_dir: Directory to save results to
        
    Returns:
        Path to the saved results file
    """
    # Normalize and ensure output directory exists
    output_dir = unicodedata.normalize('NFC', os.path.abspath(output_dir))
    os.makedirs(output_dir, exist_ok=True)
    
    # Group metrics by parameter configuration
    param_groups = {}
    for metrics in all_metrics:
        if metrics is None:
            continue
        
        # Create a key for parameter combination
        params = metrics.get('parameters', {})
        param_key = f"sf{params.get('spreading_factor', 'unknown')}_bw{params.get('bandwidth', 'unknown')}_cr{params.get('coding_rate', 'unknown')}"
        
        if param_key not in param_groups:
            param_groups[param_key] = {
                'parameter_config': params,
                'cycles': [],
                'summary': {}
            }
        
        param_groups[param_key]['cycles'].append(metrics)
    
    # Calculate summary statistics for each parameter group
    for param_key, group in param_groups.items():
        cycles = group['cycles']
        if not cycles:
            continue
        
        # Aggregate metrics across cycles for this parameter configuration
        rtts = [c.get('avg_rtt_ms') for c in cycles if c.get('avg_rtt_ms') is not None]
        ack_rssis = [c.get('avg_ack_rssi') for c in cycles if c.get('avg_ack_rssi') is not None]
        ack_snrs = [c.get('avg_ack_snr') for c in cycles if c.get('avg_ack_snr') is not None]
        pkt_rssis = [c.get('avg_packet_rssi') for c in cycles if c.get('avg_packet_rssi') is not None]
        pkt_snrs = [c.get('avg_packet_snr') for c in cycles if c.get('avg_packet_snr') is not None]
        delivery_rates = [c.get('delivery_rate') for c in cycles if c.get('delivery_rate') is not None]
        total_packets = sum(c.get('total_packets', 0) for c in cycles)
        delivered_packets = sum(c.get('delivered_packets', 0) for c in cycles)
        # Compute summary
        summary = {
            'avg_rtt_ms': sum(rtts) / len(rtts) if rtts else None,
            'avg_ack_rssi': sum(ack_rssis) / len(ack_rssis) if ack_rssis else None,
            'avg_ack_snr': sum(ack_snrs) / len(ack_snrs) if ack_snrs else None,
            'avg_packet_rssi': sum(pkt_rssis) / len(pkt_rssis) if pkt_rssis else None,
            'avg_packet_snr': sum(pkt_snrs) / len(pkt_snrs) if pkt_snrs else None,
            'avg_delivery_rate': sum(delivery_rates) / len(delivery_rates) if delivery_rates else None,
            'total_packets': total_packets,
            'delivered_packets': delivered_packets,
            'overall_delivery_rate': delivered_packets / total_packets if total_packets > 0 else 0,
            'num_cycles': len(cycles)
        }
        # --- Policy Score Calculation ---
        # Normalization bounds (domain knowledge, can be tuned):
        # RTT: 0ms (best) to 5000ms (worst, clipped)
        # RSSI: -120 (worst) to -30 (best)
        # SNR: -20 (worst) to +20 (best)
        # Delivery rate: 0 to 1
        norm_metrics = []
        # Lower RTT is better, so invert: norm = 1 - min(rtt, 5000)/5000
        if summary['avg_rtt_ms'] is not None:
            norm_metrics.append(1 - min(summary['avg_rtt_ms'], 5000) / 5000)
        # RSSI: -120 to -30, higher is better
        for k in ['avg_ack_rssi', 'avg_packet_rssi']:
            v = summary.get(k)
            if v is not None:
                norm_metrics.append((max(min(v, -30), -120) + 120) / 90)  # -120 maps to 0, -30 to 1
        # SNR: -20 to +20, higher is better
        for k in ['avg_ack_snr', 'avg_packet_snr']:
            v = summary.get(k)
            if v is not None:
                norm_metrics.append((max(min(v, 20), -20) + 20) / 40)  # -20 maps to 0, +20 to 1
        # Delivery rate: 0 to 1
        if summary['overall_delivery_rate'] is not None:
            norm_metrics.append(max(0, min(summary['overall_delivery_rate'], 1)))
        # Final policy score: mean of all normalized metrics (equal weight)
        summary['policy_score'] = sum(norm_metrics) / len(norm_metrics) if norm_metrics else None
        group['summary'] = summary
    
    # Calculate overall success metric
    all_total_packets = sum(group['summary']['total_packets'] for group in param_groups.values())
    all_delivered_packets = sum(group['summary']['delivered_packets'] for group in param_groups.values())
    overall_success = all_delivered_packets / all_total_packets if all_total_packets > 0 else 0
    # Save to file
    timestamp = int(time.time())
    # Always use repo-relative path for output
    output_file = utils.get_repo_path(os.path.join(str(node_id), f"aggregated_results_node{node_id}_{timestamp}.json"))
    # Double-check and create parent directory of output_file
    final_dir = os.path.dirname(output_file)
    if not os.path.exists(final_dir):
        try:
            os.makedirs(final_dir, exist_ok=True)
            print(f"[DEBUG] Created directory: {final_dir}")
        except Exception as e:
            logging.error(f"Failed to create directory {final_dir}: {e}")
    
    results = {
        'node_id': node_id,
        'timestamp': timestamp,
        'parameter_groups': param_groups,
        'experiment_summary': {
            'total_parameter_configs': len(param_groups),
            'total_cycles': len(all_metrics),
            'successful_cycles': len([m for m in all_metrics if m is not None]),
            'overall_success': overall_success
        }
    }
    
    logging.debug(f"Attempting to write aggregated results to: {output_file}")
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logging.info(f"Results successfully saved to {output_file}")
    except Exception as e:
        logging.error(f"Failed to write aggregated results to {output_file}: {e}")
    
    # Print summary
    for param_key, group in param_groups.items():
        summary = group['summary']
        params = group['parameter_config']
        rtt = summary['avg_rtt_ms']
        delivery = summary['overall_delivery_rate']
        rtt_str = f"{rtt:.1f}" if rtt is not None else "N/A"
        delivery_str = f"{delivery:.1%}" if delivery is not None else "N/A"
        logging.info(f"{param_key}: RTT={rtt_str}ms, Delivery={delivery_str}")
    logging.info(f"Overall success rate: {overall_success:.1%}")
    return output_file

def experiment_loop(shared_state, log_file, aggregator_mod):
    """Main experiment loop: send, evaluate, reboot, repeat.
    
    Args:
        shared_state: Dictionary containing serial_interface, mesh_iface, param_mgr, sender_mod
        log_file: Path to the log file for sent packets
        aggregator_mod: MetricAggregator module
    """
    global GLOBAL_MSG_ID
    
    import os
    import utils
    
    # Get node directory from log_file path
    node_dir = os.path.dirname(log_file)
    mesh_iface = shared_state['mesh_iface']
    node_id = mesh_iface.get_node_id()
    
    all_cycle_metrics = []
    import receiver
    import parameter_manager
    parameter_manager.param_mgr = shared_state['param_mgr']  # Make param_mgr globally accessible for receiver
    for cycle in range(8):  # Increased from 2 to 8 cycles for better learning
        # Get current LoRa parameters before sending
        mesh_iface = shared_state['mesh_iface']
        current_params = get_current_lora_params(mesh_iface)
        print_cycle_header_box(cycle+1, 8, current_params['spreading_factor'], current_params['bandwidth'], current_params['coding_rate'])
        # 1. Send broadcast packets
        serial_interface = shared_state['serial_interface']
        # If you want to use sender_mod, use shared_state['sender_mod'] here
        sent_packets_list = sender.send_broadcast_packets(
            serial_interface, duration=20, interval=2, log_file=None,
            idle_time=60, start_msg_id=GLOBAL_MSG_ID
        )
        if sent_packets_list:
            for pkt in sent_packets_list:
                SENT_PACKETS[str(pkt['id'])] = pkt
            GLOBAL_MSG_ID = sent_packets_list[-1]['id'] + 1 if sent_packets_list else GLOBAL_MSG_ID
        print("Waiting for ACKs...")
        # 2. Evaluation period - aggregate metrics for this cycle
        # Only consider packets sent in this cycle
        cycle_msg_ids = [pkt['id'] for pkt in sent_packets_list] if sent_packets_list else []
        cycle_packets = {str(pkt_id): SENT_PACKETS[str(pkt_id)] for pkt_id in cycle_msg_ids if str(pkt_id) in SENT_PACKETS}
        metrics = aggregator_mod.aggregate_from_memory(cycle_packets, receiver.ACK_METRICS)
        if metrics:
            rtt = metrics['avg_rtt_ms']
            delivery = metrics['delivery_rate']
            rtt_str = f"{rtt:.1f}" if rtt is not None else "N/A"
            delivery_str = f"{delivery:.1%}" if delivery is not None else "N/A"
            
            # Calculate overall success metric
            param_mgr = shared_state['param_mgr']
            current_params = {
                'spreading_factor': current_params.get('spread_factor', 9),
                'bandwidth': current_params.get('bandwidth', 250)
            }
            success_metric = param_mgr._calculate_reward(metrics, current_params)
            
            print_cycle_box(cycle+1, rtt_str, delivery_str, success_metric)
            
            # Annotate metrics with parameter config and cycle info
            metrics['parameters'] = current_params.copy()
            metrics['cycle'] = cycle + 1
            metrics['timestamp'] = time.time()
            all_cycle_metrics.append(metrics)
            
            # Set the last_action to the parameters actually used in this cycle
            param_mgr.last_action = {
                'spreading_factor': current_params.get('spreading_factor', 9),
                'bandwidth': current_params.get('bandwidth', 250)
            }
            param_mgr.current_metrics = metrics
            
            # ---- RL: Update contextual bandit with this cycle's performance ----
            param_mgr.record_performance(metrics)
        else:
            print(f"\033[97m")  # White color
            title = f"Cycle {cycle+1}"
            content = "No metrics available"
            width = max(len(title) + 6, len(content) + 4)  # Add padding
            
            print("┌" + "─" * (width - 2) + "┐")
            print("│ " + title.ljust(width - 4) + " │")
            print("│ " + content.ljust(width - 4) + " │")
            print("└" + "─" * (width - 2) + "┘")
            print(f"\033[0m")  # Reset color
        print("-")
        # 3. Distributed parameter update logic
        if all_cycle_metrics:
            last_metrics = all_cycle_metrics[-1]
            print_system_message("=== Parameter Evaluation ===")
            logging.info("[PARAM PHASE] Starting distributed parameter coordination...")
            
            # Evaluate all param combinations and prepare predictions
            param_mgr = shared_state['param_mgr']
            predictions = param_mgr.evaluate_all_param_combinations(last_metrics)
            mesh_iface = shared_state['mesh_iface']
            node_id = mesh_iface.get_node_id()
            
            # Wait 5 seconds to receive messages before broadcasting
            logging.info("[PARAM PHASE] Waiting 5 seconds to receive messages before broadcasting...")
            time.sleep(5)
            
            # Create quantized predictions for broadcasting to ensure consistency
            max_expected = 10.0
            quantized_predictions_for_broadcast = {}
            for param_key, score in predictions.items():
                # Apply same quantization: scale to 0-255, then back to float
                quantized_byte = max(0, min(255, int(round(score * 255 / max_expected))))
                dequantized_score = quantized_byte * max_expected / 255.0
                quantized_predictions_for_broadcast[param_key] = dequantized_score
            
            # Now broadcast quantized predictions
            param_mgr.broadcast_model_predictions(quantized_predictions_for_broadcast, node_id)

            # Wait for predictions from all nodes (including self), or until timeout
            logging.info("[PARAM PHASE] Waiting for predictions...")
            expected_node_ids = set()
            try:
                # Try to get all node IDs from the mesh interface
                if hasattr(mesh_iface.serial_interface, 'nodes') and mesh_iface.serial_interface.nodes:
                    expected_node_ids = set(mesh_iface.serial_interface.nodes.keys())
                else:
                    # Fallback: just use our own node ID
                    expected_node_ids = {str(node_id)}
            except Exception:
                expected_node_ids = {str(node_id)}
            # Always include our own node ID
            expected_node_ids.add(str(node_id))
            # Store our own predictions in predictions_received immediately
            # Apply same quantization as sent predictions for consistency
            if not hasattr(param_mgr, 'predictions_received'):
                param_mgr.predictions_received = {}
            if str(node_id) not in param_mgr.predictions_received:
                # Quantize local predictions to same precision as sent predictions
                max_expected = 10.0
                quantized_predictions = {}
                for param_key, score in predictions.items():
                    # Apply same quantization: scale to 0-255, then back to float
                    quantized_byte = max(0, min(255, int(round(score * 255 / max_expected))))
                    dequantized_score = quantized_byte * max_expected / 255.0
                    quantized_predictions[param_key] = dequantized_score
                param_mgr.predictions_received[str(node_id)] = quantized_predictions
            
            # Simple timeout-based waiting for predictions
            poll_interval = 0.5
            retransmit_interval = 5.0  # seconds
            timeout = 60.0  # seconds
            last_retransmit = time.time()
            start_time = time.time()
            
            while True:
                prediction_count = len(param_mgr.predictions_received)
                elapsed_time = time.time() - start_time
                
                # Wait for predictions from 2 nodes OR timeout after 30 seconds
                if prediction_count >= 2 or elapsed_time >= timeout:
                    break
                
                now = time.time()
                # Retransmit our quantized predictions if conditions not met and retransmit interval passed
                if now - last_retransmit > retransmit_interval:
                    param_mgr.broadcast_model_predictions(quantized_predictions_for_broadcast, node_id)
                    last_retransmit = now
                    logging.info(f"[PARAM PHASE] Retransmitting predictions. Predictions: {prediction_count}/2, Time: {elapsed_time:.1f}s/{timeout}s")
                    logging.info(f"[PARAM PHASE] param_mgr object ID: {id(param_mgr)}")
                    logging.info(f"[PARAM PHASE] predictions_received keys: {list(param_mgr.predictions_received.keys())}")
                time.sleep(poll_interval)
            
            # Log prediction collection summary
            if hasattr(param_mgr, 'predictions_received'):
                participating_nodes = list(param_mgr.predictions_received.keys())
                if prediction_count >= 2:
                    logging.info(f"[PARAM PHASE] Collection complete: {len(participating_nodes)} nodes participated")
                    logging.info(f"[PARAM PHASE] Condition met: predictions from {len(participating_nodes)} nodes")
                    
                    # Wait an additional 60 seconds while continuously rebroadcasting predictions
                    logging.info("[PARAM PHASE] Additional 30s wait with rebroadcasting...")
                    additional_wait_start = time.time()
                    additional_wait_duration = 60.0
                    rebroadcast_interval = 2.0  # Rebroadcast every 2 seconds during the additional wait
                    last_rebroadcast = time.time()
                    
                    # Create quantized predictions for rebroadcasting to ensure consistency
                    max_expected = 10.0
                    quantized_predictions_for_broadcast = {}
                    for param_key, score in predictions.items():
                        # Apply same quantization: scale to 0-255, then back to float
                        quantized_byte = max(0, min(255, int(round(score * 255 / max_expected))))
                        dequantized_score = quantized_byte * max_expected / 255.0
                        quantized_predictions_for_broadcast[param_key] = dequantized_score
                    
                    while time.time() - additional_wait_start < additional_wait_duration:
                        current_time = time.time()
                        
                        # Rebroadcast quantized predictions every second
                        if current_time - last_rebroadcast >= rebroadcast_interval:
                            param_mgr.broadcast_model_predictions(quantized_predictions_for_broadcast, node_id)
                            last_rebroadcast = current_time
                        
                        time.sleep(0.1)  # Small sleep to prevent busy waiting
                    
                    logging.info(f"[PARAM PHASE] Additional wait complete ({len(param_mgr.predictions_received)} predictions)")
                else:
                    logging.info(f"[PARAM PHASE] Timeout reached: {len(participating_nodes)} nodes participated after {timeout}s")
                    logging.info(f"[PARAM PHASE] Proceeding with available predictions from {len(participating_nodes)} nodes")
            else:
                logging.info("[PARAM PHASE] No predictions received from any nodes")
            
            # Only proceed to consensus if at least the current node participated
            # (The current node always participates by sending its own predictions)
            if hasattr(param_mgr, 'predictions_received') and len(param_mgr.predictions_received) >= 1:
                print_system_message("=== Consensus ===")
                new_params = param_mgr.decide_consensus_params()
                if new_params != current_params:
                    logging.info(f"[PARAM UPDATE] Applying: SF={new_params['spread_factor']}, BW={new_params['bandwidth']}")
                    # Save the model before reboot to ensure it's preserved
                    node_id = mesh_iface.get_node_id()
                    model_filename = f"rl_model_{node_id}.json"
                    model_path = utils.get_repo_path(os.path.join(str(node_id), model_filename))
                    try:
                        param_mgr.bandit.save_model(model_path)
                        logging.info(f"[PARAM UPDATE] Model saved ({os.path.basename(model_path)})")
                    except Exception as e:
                        logging.warning(f"[PARAM UPDATE] Failed to save model: {e}")
                    
                    print_system_message("=== Device Reboot ===")
                    param_mgr.update_params_and_reboot(new_params)
                    time.sleep(8)  # Wait for reboot (adjust as needed)
                    # Robustly reconnect to serial port after reboot
                    port_path = getattr(shared_state['serial_interface'], 'devPath', None)
                    
                    print_system_message("=== Reconnection ===")
                    
                    # Properly close the old connection
                    try:
                        if hasattr(shared_state['serial_interface'], 'close'):
                            shared_state['serial_interface'].close()
                        elif hasattr(shared_state['serial_interface'], 'stream') and hasattr(shared_state['serial_interface'].stream, 'close'):
                            shared_state['serial_interface'].stream.close()
                    except Exception as e:
                        logging.debug(f"[PARAM UPDATE] Error closing old serial connection: {e}")
                    
                    # Use the same retry logic as setup_device
                    def create_serial_interface_with_retry(port_path=None, max_retries=10):
                        """Create serial interface with retry logic for port locking issues."""
                        for attempt in range(max_retries):
                            try:
                                if port_path:
                                    return meshtastic.serial_interface.SerialInterface(devPath=port_path)
                                else:
                                    return meshtastic.serial_interface.SerialInterface()
                            except serial.serialutil.SerialException as e:
                                if "Could not exclusively lock port" in str(e) and attempt < max_retries - 1:
                                    logging.info(f"[PARAM UPDATE] Serial port locked, retrying in 2 seconds (attempt {attempt + 1}/{max_retries})...")
                                    time.sleep(2)
                                    continue
                                else:
                                    raise e
                    
                    logging.info("[PARAM UPDATE] Reconnecting to device...")
                    serial_interface = create_serial_interface_with_retry(port_path)
                    time.sleep(3)
                    mesh_iface = mesh.MeshInterface(param_mgr.config, serial_interface)
                    # Recreate param_mgr and sender_mod with new interfaces
                    # Preserve the existing bandit model by passing it to the new coordinator
                    old_bandit = param_mgr.bandit
                    param_mgr = parameter_manager.ParameterCoordinator(param_mgr.config, mesh_iface)
                    # Restore the existing bandit model to avoid reinitialization
                    param_mgr.bandit = old_bandit
                    logging.info("[PARAM UPDATE] RL model restored")
                    sender_mod = sender.PacketGenerator(param_mgr.config, serial_interface, log_file)
                    shared_state['serial_interface'] = serial_interface
                    shared_state['mesh_iface'] = mesh_iface
                    shared_state['param_mgr'] = param_mgr
                    shared_state['sender_mod'] = sender_mod
                    # Update the global param_mgr for receiver access
                    parameter_manager.param_mgr = param_mgr
                    logging.info(f"[PARAM UPDATE] Updated global param_mgr reference: {id(param_mgr)}")
                    # Ensure the message handler uses the latest serial interface after reboot
                    handle_incoming_message.serial_interface = serial_interface
                    # Clear any stale predictions from before reboot
                    if hasattr(param_mgr, 'predictions_received'):
                        param_mgr.predictions_received.clear()
                    logging.info("[PARAM UPDATE] Cleared stale predictions after reboot")
                    node_id = mesh_iface.get_node_id()
                    logging.info(f"[PARAM UPDATE] Device ready (Node: {node_id})")
                    # Re-subscribe to message handler after reconnect
                    pub.subscribe(handle_incoming_message, "meshtastic.receive")
                    # Send a broadcast READY message
                    ready_msg = f"READY_{node_id}"
                    try:
                        serial_interface.sendData(ready_msg.encode('utf-8'))
                        logging.info(f"[SYNC] Broadcasted readiness message: {ready_msg}")
                    except Exception as e:
                        logging.warning(f"[SYNC] Failed to broadcast readiness message: {e}")
                    print_system_message("=== Synchronization ===")
                    logging.info("[SYNC] Waiting 15 seconds for all nodes to be ready after parameter update and reboot...")
                    time.sleep(15)
                else:
                    logging.info("[PARAM UPDATE] No parameter change needed.")
            else:
                # Add debug logging to see what's in predictions_received
                if hasattr(param_mgr, 'predictions_received'):
                    logging.info(f"[PARAM DEBUG] predictions_received keys: {list(param_mgr.predictions_received.keys())}")
                    logging.info(f"[PARAM DEBUG] predictions_received count: {len(param_mgr.predictions_received)}")
                else:
                    logging.info("[PARAM DEBUG] predictions_received attribute does not exist")
                logging.info("[PARAM UPDATE] Skipping parameter update: no predictions available.")
        else:
            logging.info("[PARAM PHASE] No metrics available for parameter update.")

    # Final aggregation across all cycles
    logging.info("\n[AGGREGATE] Final aggregation across all cycles...")
    save_aggregated_results(all_cycle_metrics, node_id, output_dir=node_dir)

    # Save all sent packets to a single JSON file
    import json
    sent_packets_path = utils.get_repo_path(os.path.join(str(node_id), 'sent_packets_all.json'))
    # Ensure the directory exists before writing
    sent_packets_dir = os.path.dirname(sent_packets_path)
    os.makedirs(sent_packets_dir, exist_ok=True)
    with open(sent_packets_path, 'w') as f:
        json.dump(list(SENT_PACKETS.values()), f, indent=2)
    logging.info(f"All sent packets saved to {sent_packets_path}")

    logging.info("\n[COMPLETE] Experiment cycles finished.")

# =====================
# Device Setup
# =====================
def setup_device(cfg, args):
    """Initialize the serial interface and mesh interface.
    
    Args:
        cfg: Configuration object
        args: Command line arguments
        
    Returns:
        Tuple of (serial_interface, mesh_iface)
    """
    def create_serial_interface(port_path=None, max_retries=5):
        """Create serial interface with retry logic for port locking issues."""
        for attempt in range(max_retries):
            try:
                if port_path:
                    return meshtastic.serial_interface.SerialInterface(devPath=port_path)
                else:
                    return meshtastic.serial_interface.SerialInterface()
            except serial.serialutil.SerialException as e:
                if "Could not exclusively lock port" in str(e) and attempt < max_retries - 1:
                    logging.warning(f"Serial port locked, retrying in 2 seconds (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(2)
                    continue
        # If we get here, all retries failed
        raise serial.serialutil.SerialException("Failed to create serial interface after all retries")
    
    # Initial connection
    serial_interface = create_serial_interface(args.port)
    logging.info("Serial interface created. Waiting for device boot...")
    time.sleep(3)
    logging.info("Device boot wait complete. Proceeding with initialization.")
    mesh_iface = mesh.MeshInterface(cfg, serial_interface)
    node_id = mesh_iface.get_node_id()
    
    rebooted = mesh_iface.ensure_lora_params()
    if rebooted:
        logging.info("Waiting for device to reboot and reconnect...")
        
        # Properly close the old connection
        try:
            if hasattr(serial_interface, 'close'):
                serial_interface.close()
            elif hasattr(serial_interface, 'stream') and hasattr(serial_interface.stream, 'close'):
                serial_interface.stream.close()
        except Exception as e:
            logging.debug(f"Error closing old serial connection: {e}")
        
        time.sleep(8)  # Wait for reboot
        
        # Re-initialize with retry logic
        logging.info("Attempting to reconnect to serial port after reboot...")
        serial_interface = create_serial_interface(args.port)
        logging.info("Serial interface reconnected after reboot. Waiting for device boot...")
        time.sleep(3)
        mesh_iface = mesh.MeshInterface(cfg, serial_interface)
        node_id = mesh_iface.get_node_id()
        logging.info("Device ready after reboot.")
    
    return serial_interface, mesh_iface

def print_node_keys(mesh_iface):
    """Print public and private keys for the local node.
    
    Args:
        mesh_iface: Meshtastic mesh interface object
    """
    node = mesh_iface.get_local_node()
    if not node:
        return
    # Try to access security config
    security_config = getattr(node.localConfig, 'security', None)
    if not security_config:
        return
    keys = {}
    if hasattr(security_config, 'publicKey') and security_config.publicKey:
        keys['publicKey'] = base64.b64encode(security_config.publicKey).decode('utf-8')
        logging.info(f"Public Key: {keys['publicKey']}")
    if hasattr(security_config, 'privateKey') and security_config.privateKey:
        keys['privateKey'] = base64.b64encode(security_config.privateKey).decode('utf-8')
        logging.info(f"Private Key: {keys['privateKey']}")
    return keys

def save_node_config(mesh_iface):
    """Save the local node configuration to a file.
    
    Args:
        mesh_iface: Meshtastic mesh interface object
    """
    node = mesh_iface.get_local_node()
    if not node:
        return
    config_obj = node.localConfig
    from google.protobuf.json_format import MessageToDict
    config_dict = MessageToDict(config_obj, preserving_proto_field_name=True)
    logging.info(f"[INFO] Local node config loaded.")
    # No file saving here

PRIVATE_KEY_B64 = None

def extract_private_key_from_config(cfg):
    """Extract private key from configuration file.
    
    Args:
        cfg: Configuration object
    """
    global PRIVATE_KEY_B64
    logging.debug(f"[DEBUG] Config type: {type(cfg)}")
    logging.debug(f"[DEBUG] Config keys/attrs: {list(cfg.keys()) if isinstance(cfg, dict) else dir(cfg)}")
    sec = cfg.get('security', {}) if isinstance(cfg, dict) else getattr(cfg, 'security', None)
    logging.debug(f"[DEBUG] Security section: {sec}")
    if isinstance(sec, dict):
        private_key = sec.get('private_key')
    elif sec:
        private_key = getattr(sec, 'private_key', None)
    else:
        private_key = None
    if private_key:
        PRIVATE_KEY_B64 = private_key
        logging.info(f"[INFO] Extracted private key from config: {PRIVATE_KEY_B64}")
    else:
        logging.warning("No private key found in config.")

def extract_private_key_from_device(mesh_iface):
    """Extract private key from the running device.
    
    Args:
        mesh_iface: Meshtastic mesh interface object
    """
    global PRIVATE_KEY_B64
    node = mesh_iface.get_local_node()
    if not node:
        return
    config_obj = node.localConfig
    from google.protobuf.json_format import MessageToDict
    config_dict = MessageToDict(config_obj, preserving_proto_field_name=True)
    sec = config_dict.get('security', {})
    private_key = sec.get('private_key')
    if private_key:
        PRIVATE_KEY_B64 = private_key

# =====================
# Main Entry Point
# =====================
def main():
    """Main entry point for the Adaptive Mesh Network Controller."""
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Adaptive Mesh Network Controller')
    parser.add_argument('--config', type=str, help='Path to config file', default='config.json')
    parser.add_argument('--port', type=str, help='USB serial port for Meshtastic device', default=None)
    args = parser.parse_args()

    # --- Configuration ---
    user_specified_config = args.config != parser.get_default('config')
    cfg = config.load_config(args.config, user_specified=user_specified_config)
    # Overwrite config values with CLI arguments if provided
    cli_overrides = {}
    if args.port is not None:
        cli_overrides['port'] = args.port
    if cli_overrides:
        config.merge_config(cfg, cli_overrides)
    # Extract private key for later decryption
    # extract_private_key_from_config(cfg) # Removed as per edit hint
    run_id = random.randint(10000, 99999)
    log_file = f'sent_packets_{run_id}.json'

    # --- Device Setup ---
    serial_interface, mesh_iface = setup_device(cfg, args)
    
    # --- Display consolidated node information immediately ---
    node_id = mesh_iface.get_node_id()
    save_node_config(mesh_iface)
    extract_private_key_from_device(mesh_iface)
    
    # Create consolidated information box (wider to accommodate private key)
    node_info = f"Port: {args.port or 'auto-detect'}\nNode ID: !{node_id:08x}\nDecimal: {node_id}\nPrivate Key: {PRIVATE_KEY_B64 if PRIVATE_KEY_B64 else 'Not available'}"
    print_info_box("NODE INFORMATION", node_info, width=80)
    
    print_node_keys(mesh_iface)
    handle_incoming_message.serial_interface = serial_interface
    logging.debug(f"handle_incoming_message serial: {serial_interface}")
    global NODE_ID_TO_PUBKEY
    NODE_ID_TO_PUBKEY = get_all_node_public_keys(serial_interface)

    # --- Log file naming logic ---
    node_dir = unicodedata.normalize('NFC', str(node_id))
    node_dir = os.path.abspath(node_dir)
    if not os.path.exists(node_dir):
        os.makedirs(node_dir, exist_ok=True)
    # Find next available integer for sent_packetsN.json
    existing = [f for f in os.listdir(node_dir) if f.startswith('sent_packets') and f.endswith('.json')]
    nums = []
    for f in existing:
        try:
            n = int(f.replace('sent_packets', '').replace('.json', ''))
            nums.append(n)
        except Exception:
            pass
    next_num = max(nums) + 1 if nums else 1
    log_file = os.path.join(node_dir, f'sent_packets{next_num}.json')

    # Set the sent log file for receiver
    import utils
    log_file_full_path = utils.get_repo_path(log_file)

    # --- Module Initialization ---
    print_green_section_header("MODULE INITIALIZATION")
    sender_mod = sender.PacketGenerator(cfg, serial_interface, log_file)
    aggregator_mod = aggregator.MetricAggregator(cfg)
    param_mgr = parameter_manager.ParameterCoordinator(cfg, mesh_iface)
    
    # Set global param_mgr reference for receiver access
    parameter_manager.param_mgr = param_mgr
    logging.info(f"[INIT] Set global param_mgr reference: {id(param_mgr)}")

    # --- PubSub subscription for receiving messages ---
    pub.subscribe(handle_incoming_message, "meshtastic.receive")

    # --- Shared state for interfaces and managers ---
    shared_state = {
        'serial_interface': serial_interface,
        'mesh_iface': mesh_iface,
        'param_mgr': param_mgr,
        'sender_mod': sender_mod
    }

    # --- Experiment Execution ---
    print_yellow_section_header("EXPERIMENT START")
    experiment_thread = threading.Thread(
        target=experiment_loop,
        args=(shared_state, log_file, aggregator_mod),
        daemon=True
    )
    experiment_thread.start()

    # --- Main Loop: Keep Alive for PubSub/Receiving ---
    try:
        while experiment_thread.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("\n[INFO] Interrupted by user. Exiting...")

    # At the end of the experiment, write SENT_PACKETS to disk
    with open('experiment_results.json', 'w') as f:
        json.dump(list(SENT_PACKETS.values()), f, indent=2)

if __name__ == '__main__':
    main()
