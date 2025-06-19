import utils
import json
import glob
import os
import time
import re
import logging

try:
    from main import MSG_ID_TO_LOGFILE
except ImportError:
    MSG_ID_TO_LOGFILE = {}

SENT_PACKETS = {}
ACK_METRICS = {}

def get_sent_packets():
    """Get the current SENT_PACKETS from main module, or fallback to local.
    
    Returns:
        Dictionary containing sent packet information
    """
    try:
        import main
        return main.SENT_PACKETS
    except (ImportError, AttributeError):
        return SENT_PACKETS

class PacketReceiver:
    """Handles packet reception, filtering, and special payload processing."""
    
    def __init__(self, config=None, serial_interface=None):
        """Initialize the packet receiver.
        
        Args:
            config: Configuration dictionary (optional)
            serial_interface: Serial interface object (optional)
        """
        self.config = config
        self.serial_interface = serial_interface

    @staticmethod
    def filter_and_format_packet(packet):
        """Filter and format a received packet for display.
        
        Args:
            packet: Dictionary containing packet data
            
        Returns:
            Tuple of (msg_type, display_text) where msg_type is 'text', 'payload', 'ack', or 'unknown'
        """
        decoded = packet.get('decoded')
        if decoded:
            # Check for MSG_ or ACK_ in decoded payload (bytes or str)
            payload = decoded.get('payload')
            if payload is not None:
                if isinstance(payload, bytes):
                    try:
                        payload_str = payload.decode('utf-8', errors='replace')
                    except Exception:
                        payload_str = str(payload)
                else:
                    payload_str = str(payload)
                if payload_str.startswith('MSG_'):
                    return ('payload', payload_str)
                if payload_str.startswith('ACK_'):
                    return ('ack', payload_str)
                if payload_str.startswith('SUCCESS_PREDICTIONS_FROM_'):
                    return ('payload', payload_str)
            # If not MSG_ or ACK_, check for text
            text = decoded.get('text')
            if text:
                if text.startswith('ACK_'):
                    return ('ack', text)
                if text.startswith('MSG_'):
                    return ('payload', text)
                if text.startswith('SUCCESS_PREDICTIONS_FROM_'):
                    return ('payload', text)
                return ('text', text)
            # Fallback: show payload as text if possible
            if payload is not None:
                if payload_str.startswith('ACK_'):
                    return ('ack', payload_str)
                if payload_str.startswith('MSG_'):
                    return ('payload', payload_str)
                if payload_str.startswith('SUCCESS_PREDICTIONS_FROM_'):
                    return ('payload', payload_str)
                return ('text', payload_str)
        # Fallback: raw payload
        raw_payload = packet.get('payload')
        if raw_payload:
            if isinstance(raw_payload, bytes):
                try:
                    raw_str = raw_payload.decode('utf-8', errors='replace')
                except Exception:
                    raw_str = str(raw_payload)
            else:
                raw_str = str(raw_payload)
            if raw_str.startswith('MSG_'):
                return ('payload', raw_str)
            if raw_str.startswith('ACK_'):
                return ('ack', raw_str)
            if raw_str.startswith('SUCCESS_PREDICTIONS_FROM_'):
                return ('payload', raw_str)
            return ('text', raw_str)
        # Fallback: scan all string fields for ACK_ or MSG_
        for v in packet.values():
            if isinstance(v, str):
                if v.startswith('ACK_'):
                    return ('ack', v)
                if v.startswith('MSG_'):
                    return ('payload', v)
                if v.startswith('SUCCESS_PREDICTIONS_FROM_'):
                    return ('payload', v)
        return ('unknown', str(packet))

    @staticmethod
    def send_text_packet(serial_interface, text, destination_id=None):
        """Send a text message as UTF-8 using sendData.
        
        Args:
            serial_interface: Serial interface object
            text: Text message to send
            destination_id: Destination node ID (optional, always broadcasts)
            
        Returns:
            True if sent successfully, False otherwise
        """
        if serial_interface and text:
            # Always send as broadcast (no destinationId), for both MSG and ACK
            serial_interface.sendData(text.encode('utf-8'))
            return True
        return False

    @staticmethod
    def send_ack_packet(serial_interface, ack_payload, destination_id):
        """Send an ACK packet as a text message.
        
        Args:
            serial_interface: Serial interface object
            ack_payload: ACK payload text
            destination_id: Destination node ID (ignored, always broadcasts)
            
        Returns:
            True if sent successfully, False otherwise
        """
        # Ignore destination_id, always broadcast
        return PacketReceiver.send_text_packet(serial_interface, ack_payload, None)

    @staticmethod
    def send_prediction_success_message(serial_interface, sender_node_id):
        """Send a success message to acknowledge receipt of predictions.
        
        Args:
            serial_interface: Serial interface object
            sender_node_id: ID of the node that sent predictions
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not serial_interface:
            return False
        try:
            # Create success message: SUCCESS_PREDICTIONS_FROM_{sender_node_id}
            success_payload = f"SUCCESS_PREDICTIONS_FROM_{sender_node_id}"
            return PacketReceiver.send_text_packet(serial_interface, success_payload, None)
        except Exception as e:
            logging.warning(f"[SUCCESS SEND] Failed to send success message: {e}")
            return False

    @staticmethod
    def handle_model_prediction_packet(packet, serial_interface=None):
        """Parse incoming prediction messages from other nodes.
        
        Args:
            packet: Dictionary containing packet data
            serial_interface: Serial interface object (optional)
            
        Returns:
            True if prediction was processed successfully, False otherwise
        """
        import logging
        try:
            import parameter_manager
            param_mgr = getattr(parameter_manager, 'param_mgr', None)
            if not param_mgr:
                logging.debug("[PARAM RECEIVE] No param_mgr available")
                return False
            
            # Add detailed debug logging
            logging.debug(f"[PARAM RECEIVE] Processing packet: {packet.get('fromId', 'unknown')} -> {packet.get('toId', 'broadcast')}")
            
            decoded = packet.get('decoded')
            payload = None
            if decoded and 'payload' in decoded:
                payload = decoded['payload']
                if isinstance(payload, bytes):
                    try:
                        payload = payload
                    except Exception:
                        payload = decoded['payload']
                else:
                    payload = decoded['payload']
            elif 'payload' in packet:
                payload = packet['payload']
                if isinstance(payload, bytes):
                    try:
                        payload = payload
                    except Exception:
                        payload = packet['payload']
                else:
                    payload = packet['payload']
            
            if payload:
                logging.debug(f"[PARAM RECEIVE] Processing payload: type={type(payload)}, length={len(payload) if hasattr(payload, '__len__') else 'N/A'}")
                
                # --- New binary format: b'PRED' + node_id (8 bytes) + N scores (1 byte each) ---
                if isinstance(payload, bytes) and payload.startswith(b'PRED'):
                    # Only log at debug level to reduce verbosity
                    logging.debug(f"[PARAM RECEIVE] Binary PRED: {len(payload)} bytes")
                    try:
                        node_id_bytes = payload[4:12]
                        node_id = node_id_bytes.decode('utf-8', errors='replace').rstrip('_')
                        scores_bytes = payload[12:]
                        param_list = sorted([(sf, bw) for sf in param_mgr.param_ranges['spreading_factor'] for bw in param_mgr.param_ranges['bandwidth']])
                        if len(scores_bytes) != len(param_list):
                            logging.warning(f"[PARAM RECEIVE] Mismatch in received scores count: {len(scores_bytes)} vs expected {len(param_list)} (binary mode)")
                            return False
                        # Dequantize: scale back from 0-255 to 0-max_expected range
                        max_expected = 10.0
                        scores = [b * max_expected / 255.0 for b in scores_bytes]
                        preds_dict = {param_list[i]: scores[i] for i in range(len(scores))}
                        if not hasattr(param_mgr, 'predictions_received'):
                            param_mgr.predictions_received = {}
                        if node_id not in param_mgr.predictions_received:
                            param_mgr.predictions_received[node_id] = preds_dict
                            logging.info(f"[PARAM RECEIVE] NEW: Node {node_id} ({len(preds_dict)} scores)")
                            # No success message sent - just store the predictions
                        else:
                            # Only log at debug level for duplicates, and make it clear it's a duplicate
                            logging.debug(f"[PARAM RECEIVE] DUPLICATE: Skipping repeated predictions from node {node_id}")
                        return True
                    except Exception as e:
                        logging.warning(f"[PARAM RECEIVE] Failed to parse binary prediction message: {e}")
                        import traceback
                        logging.debug(f"[PARAM RECEIVE] Traceback: {traceback.format_exc()}")
                        return False
                # --- Old JSON format: 'PRED_' + json ---
                if isinstance(payload, str) and payload.startswith('PRED_'):
                    # Only log at debug level to reduce verbosity
                    logging.debug(f"[PARAM RECEIVE] String PRED: {payload[:30]}...")
                    import json
                    try:
                        msg = json.loads(payload[5:])
                        if msg.get('type') == 'PARAM_PREDICTIONS':
                            node_id = msg.get('node_id')
                            scores = msg.get('scores')
                            param_list = sorted([(sf, bw) for sf in param_mgr.param_ranges['spreading_factor'] for bw in param_mgr.param_ranges['bandwidth']])
                            if len(scores) != len(param_list):
                                logging.warning(f"[PARAM RECEIVE] Mismatch in received scores count: {len(scores)} vs expected {len(param_list)} (json mode)")
                                return False
                            preds_dict = {param_list[i]: scores[i] for i in range(len(scores))}
                            if not hasattr(param_mgr, 'predictions_received'):
                                param_mgr.predictions_received = {}
                            if node_id not in param_mgr.predictions_received:
                                param_mgr.predictions_received[node_id] = preds_dict
                                logging.info(f"[PARAM RECEIVE] NEW: Node {node_id} ({len(preds_dict)} scores)")
                                # No success message sent - just store the predictions
                            else:
                                # Only log at debug level for duplicates, and make it clear it's a duplicate
                                logging.debug(f"[PARAM RECEIVE] DUPLICATE: Skipping repeated predictions from node {node_id}")
                            return True
                    except Exception as e:
                        logging.warning(f"[PARAM RECEIVE] Failed to parse prediction message: {e}")
                        import traceback
                        logging.debug(f"[PARAM RECEIVE] Traceback: {traceback.format_exc()}")
                        return False
                else:
                    logging.debug(f"[PARAM RECEIVE] Payload does not match PRED format: {payload[:20] if hasattr(payload, '__getitem__') else str(payload)[:20]}")
            else:
                logging.debug("[PARAM RECEIVE] No payload found in packet")
        except Exception as e:
            logging.warning(f"[PARAM RECEIVE] Error handling prediction packet: {e}")
            import traceback
            logging.debug(f"[PARAM RECEIVE] Traceback: {traceback.format_exc()}")
        return False

    @staticmethod
    def handle_special_payload(packet, serial_interface=None):
        """Handle MSG_ and ACK_ payloads for ACK sending and RTT calculation.
        
        Args:
            packet: Dictionary containing packet data
            serial_interface: Serial interface object (optional)
            
        Returns:
            String with extra info (e.g., ACK sent, RTT, etc.) or None if not applicable
        """
        # Distributed parameter coordination: try to receive model predictions
        PacketReceiver.handle_model_prediction_packet(packet, serial_interface)
        
        # Extract payload first
        decoded = packet.get('decoded')
        payload = None
        if decoded and 'payload' in decoded:
            payload = decoded['payload']
        elif 'payload' in packet:
            payload = packet['payload']
        
        if not payload:
            return None
        
        # Convert payload to string if it's bytes
        if isinstance(payload, bytes):
            try:
                payload = payload.decode('utf-8', errors='replace')
            except Exception as e:
                logging.warning(f"[SPECIAL PAYLOAD] Failed to decode payload: {e}")
                return None
        
        # MSG_ logic: send ACK
        if payload.startswith('MSG_'):
            parts = payload.split('_')
            if len(parts) >= 3 and serial_interface is not None:
                msg_id = parts[1]
                rssi = packet.get('rxRssi')
                snr = packet.get('rxSnr')
                sender_id = packet.get('fromId')
                if sender_id:
                    ack_payload = f"ACK_{msg_id}_TO_{sender_id}_RSSI_{rssi}_SNR_{snr}"
                    sent = PacketReceiver.send_ack_packet(serial_interface, ack_payload, sender_id)
                    if sent:
                        return f"[ACK SENT] {ack_payload} to {sender_id}"
                    else:
                        return f"[ERROR] Failed to send ACK to {sender_id}"
                else:
                    return f"[ERROR] No sender_id found for MSG_ packet: {payload}"
        # ACK_ logic: calculate RTT, but only if this node is the intended recipient
        elif payload.startswith('ACK_'):
            parts = payload.split('_')
            if len(parts) >= 8:
                msg_id = parts[1]
                recipient_id = parts[3]
                # Ensure ack_rssi and ack_snr are floats
                try:
                    ack_rssi = float(parts[5])
                except Exception:
                    ack_rssi = None
                try:
                    ack_snr = float(parts[7])
                except Exception:
                    ack_snr = None
                # Get our own node ID using the proper method
                try:
                    if serial_interface:
                        node_info = serial_interface.getMyNodeInfo()
                        if node_info:
                            my_id = f"!{node_info.get('num'):08x}"
                        else:
                            local_node = serial_interface.localNode
                            my_id = str(getattr(local_node, 'id', None) or getattr(local_node, 'nodeNum', None))
                    else:
                        my_id = None
                except Exception as e:
                    print(f"[ERROR] Failed to get node info: {e}")
                    my_id = None
                if recipient_id != my_id:
                    return None  # Ignore ACKs not intended for us
                packet_rssi = packet.get('rxRssi')
                packet_snr = packet.get('rxSnr')
                send_time, _ = PacketReceiver.get_send_time_for_msg_id(msg_id)
                rtt_ms = None
                ack_metrics = {
                    'ack_rssi': ack_rssi,
                    'ack_snr': ack_snr,
                    'packet_rssi': packet_rssi,
                    'packet_snr': packet_snr
                }
                if send_time is not None:
                    rtt_ms = (utils.current_time() - send_time) * 1000
                    print(f"ACKed:  MSG_{msg_id} <- ACK_{msg_id} (RTT: {rtt_ms:.1f} ms, RSSI: {ack_rssi}, SNR: {ack_snr})")
                    # Update the corresponding sent packet in memory with RTT and ACK metrics
                    sent_packets = get_sent_packets()
                    pkt = sent_packets.get(str(msg_id))
                    if pkt is not None:
                        pkt['rtt_ms'] = rtt_ms
                        pkt['ack_rssi'] = ack_rssi
                        pkt['ack_snr'] = ack_snr
                        pkt['packet_rssi'] = packet_rssi
                        pkt['packet_snr'] = packet_snr
                else:
                    print(f"ACKed:  MSG_{msg_id} <- ACK_{msg_id} (RTT: UNKNOWN, RSSI: {ack_rssi}, SNR: {ack_snr})")
                # Store ACK metrics in memory for later aggregation
                ack_metrics['rtt_ms'] = rtt_ms
                ack_metrics['msg_id'] = msg_id
                ACK_METRICS[msg_id] = ack_metrics
                return None
        return None

    @staticmethod
    def get_send_time_for_msg_id(msg_id):
        """Get the send time for a specific message ID.
        
        Args:
            msg_id: Message identifier
            
        Returns:
            Tuple of (send_time, None) or (None, None) if not found
        """
        msg_id_str = str(msg_id)
        sent_packets = get_sent_packets()
        pkt = sent_packets.get(msg_id_str)
        if pkt:
            return pkt.get('timestamp'), None
        else:
            logging.warning(f"No in-memory packet found for msg_id={msg_id}")
            return None, None
