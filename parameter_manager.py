import mesh
import random
import logging
from rl import ContextualBandit

class ParameterCoordinator:
    """Manages LoRa parameter selection using contextual bandit reinforcement learning."""
    
    def __init__(self, config, mesh_iface):
        """Initialize the parameter coordinator.
        
        Args:
            config: Configuration dictionary containing RL parameters
            mesh_iface: Meshtastic mesh interface object
        """
        self.config = config
        self.mesh = mesh_iface
        # Define available parameter ranges for LoRa (remove coding_rate)
        self.param_ranges = {
            'spreading_factor': [7, 8, 9, 10, 11, 12],  # SF7 to SF12
            'bandwidth': [125, 250, 500],  # 125kHz, 250kHz, 500kHz
        }
        # Track parameter history for learning
        self.param_history = []
        self.performance_history = []
        
        # Initialize contextual bandit for ML-based parameter selection
        self.bandit = ContextualBandit(config)
        self.current_metrics = {}
        self.last_action = None
        
        # Try to load existing model with node-specific filename
        import os
        import utils
        node_id = None
        try:
            node_id = self.mesh.get_node_id()
        except Exception as e:
            node_id = 'unknown'
            logging.warning(f"[PARAM] Failed to get node_id: {e}, using 'unknown'")
        
        model_filename = f"rl_model_{node_id}.json"
        model_path = utils.get_repo_path(os.path.join(str(node_id), model_filename))
        
        try:
            self.bandit.load_model(model_path)
            logging.info(f"[PARAM] Loaded existing RL model from {model_path}")
            # Log model state after loading
            logging.info(f"[PARAM] Model state after loading: {len(self.bandit.performance_history)} performance records")
        except FileNotFoundError:
            logging.info(f"[PARAM] No existing RL model found at {model_path}, initializing with minimal priors")
            self._initialize_minimal_model()
        except Exception as e:
            logging.warning(f"[PARAM] Failed to load existing RL model from {model_path}: {e}, initializing with minimal priors")
            self._initialize_minimal_model()

    def decide_new_params(self, metrics):
        """Decide new LoRa parameters using contextual bandit.
        
        Args:
            metrics: Dictionary containing current network performance metrics
            
        Returns:
            Dictionary containing selected spreading factor and bandwidth
        """
        self.current_metrics = metrics
        
        # Use contextual bandit to select parameters
        if metrics:
            # Add current parameters to context
            context = metrics.copy()
            context['parameters'] = {
                'spreading_factor': metrics.get('parameters', {}).get('spreading_factor', 9),
                'bandwidth': metrics.get('parameters', {}).get('bandwidth', 250)
            }
            
            # Select action using bandit
            action = self.bandit.select_action(context)
            self.last_action = action
            
            logging.info(f"[PARAM] Bandit selected SF={action['spreading_factor']}, BW={action['bandwidth']}")
            logging.debug(f"[PARAM] Selected action details: {action}")
            return action
        else:
            # Fallback to random selection if no metrics
            return self.random_params()

    def random_params(self):
        """Generate random LoRa parameters.
        
        Returns:
            Dictionary containing random spreading factor and bandwidth
        """
        return {
            'spread_factor': random.choice(self.param_ranges['spreading_factor']),
            'bandwidth': random.choice(self.param_ranges['bandwidth'])
        }

    def update_params_and_reboot(self, new_params):
        """Apply new LoRa parameters and reboot the device if needed.
        
        Args:
            new_params: Dictionary containing new LoRa parameters
        """
        self.mesh.apply_new_params(new_params)
        self.param_history.append(new_params)

    def record_performance(self, metrics):
        """Record performance metrics and update the bandit.
        
        Args:
            metrics: Dictionary containing performance metrics
        """
        # logging.info(f"[PARAM] record_performance called with metrics: {list(metrics.keys()) if metrics else 'None'}")
        logging.info(f"[PARAM] Performance history length before: {len(self.performance_history)}")
        
        self.performance_history.append(metrics)
        
        # Update bandit with reward if we have a previous action
        if self.last_action and self.current_metrics:
            logging.info(f"[PARAM] Updating bandit with last_action: {self.last_action}")
            
            # Calculate reward based on performance
            reward = self._calculate_reward(metrics, self.last_action)
            
            # Log the overall success metric
            logging.info(f"[PARAM] Overall success metric (reward): {reward:.3f} for SF={self.last_action['spreading_factor']}, BW={self.last_action['bandwidth']}")
            
            # Update bandit
            context = self.current_metrics.copy()
            context['parameters'] = {
                'spreading_factor': self.current_metrics.get('parameters', {}).get('spreading_factor', 9),
                'bandwidth': self.current_metrics.get('parameters', {}).get('bandwidth', 250)
            }
            
            # Log model state before update
            logging.debug(f"[PARAM] Model state before update: {len(self.bandit.performance_history)} performance records")
            
            self.bandit.update_policy(reward, self.last_action, context)
            
            # Log model state after update
            logging.debug(f"[PARAM] Model state after update: {len(self.bandit.performance_history)} performance records")
            logging.debug(f"[PARAM] Updated bandit with reward {reward:.3f}")
            
            # Save model after every evaluation using absolute repo path with node ID
            import os
            import utils
            node_id = None
            try:
                node_id = self.mesh.get_node_id()
            except Exception as e:
                node_id = 'unknown'
                logging.warning(f"[PARAM] Failed to get node_id: {e}, using 'unknown'")
            
            model_filename = f"rl_model_{node_id}.json"
            model_path = utils.get_repo_path(os.path.join(str(node_id), model_filename))
            
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                
                # Save the model
                self.bandit.save_model(model_path)
                
                # Verify the file was actually created
                if os.path.exists(model_path):
                    file_size = os.path.getsize(model_path)
                    logging.info(f"[PARAM] Model saved: {model_filename} ({file_size} bytes) after {len(self.performance_history)} evaluations")
                else:
                    logging.error(f"[PARAM] Model save appeared successful but file does not exist: {model_path}")
                    
            except Exception as e:
                logging.error(f"[PARAM] Failed to save model to {model_path}: {e}")
                import traceback
                logging.error(f"[PARAM] Full traceback: {traceback.format_exc()}")
        else:
            logging.info(f"[PARAM] Skipping bandit update - missing last_action or current_metrics")
            logging.info(f"[PARAM]   last_action: {self.last_action}")
            logging.info(f"[PARAM]   current_metrics: {self.current_metrics is not None}")

    def _calculate_reward(self, metrics, action_params):
        """Calculate reward based on performance metrics.
        
        Args:
            metrics: Dictionary containing performance metrics
            action_params: Dictionary containing action parameters
            
        Returns:
            Float reward value based on delivery rate, latency, and energy efficiency
        """
        delivery_rate = metrics.get('delivery_rate', 1.0)
        avg_rtt = metrics.get('avg_rtt', 1000.0)
        energy_efficiency = metrics.get('energy_efficiency', 1.0)
        
        # Reward components
        delivery_reward = delivery_rate * 0.5  # 50% weight on delivery
        
        # Latency reward (lower RTT is better)
        max_rtt = 5000.0
        latency_reward = max(0, (max_rtt - avg_rtt) / max_rtt) * 0.3  # 30% weight
        
        # Energy efficiency reward
        energy_reward = energy_efficiency * 0.2  # 20% weight
        
        total_reward = delivery_reward + latency_reward + energy_reward
        
        return total_reward

    def _format_predictions_box(self, predictions, top_n=18):
        """Create a formatted box showing all parameter predictions.
        
        Args:
            predictions: Dictionary mapping parameter tuples to scores
            top_n: Number of top predictions to display (default: 18)
            
        Returns:
            Formatted string containing the predictions box
        """
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        all_predictions = sorted_predictions[:top_n]  # Show all 18 combinations
        
        # Create the box content
        lines = []
        lines.append("┌─ All Parameter Predictions ─┐")
        for i, ((sf, bw), score) in enumerate(all_predictions, 1):
            # Add proper spacing after the score to align the right border
            lines.append(f"│ {i:2d}. SF={sf:2d}, BW={bw:3d}: {score:.3f}    │")
        lines.append("└─────────────────────────────┘")
        
        return "\n".join(lines)

    def evaluate_all_param_combinations(self, metrics):
        """Use contextual bandit to predict performance for all parameter combinations.
        
        Args:
            metrics: Dictionary containing current network metrics
            
        Returns:
            Dictionary mapping parameter tuples to predicted scores
        """
        # Reset broadcast logging flag for new evaluation cycle
        if hasattr(self, '_broadcast_logged'):
            delattr(self, '_broadcast_logged')
            
        if not metrics:
            # Fallback to random predictions if no metrics
            import random
            results = {}
            for sf in self.param_ranges['spreading_factor']:
                for bw in self.param_ranges['bandwidth']:
                    score = random.random()
                    results[(sf, bw)] = score
            logging.info(f"[PARAM EVAL] Generated {len(results)} random parameter predictions.")
            return results
        
        # Use bandit to get predictions for all arms
        context = metrics.copy()
        context['parameters'] = {
            'spreading_factor': metrics.get('parameters', {}).get('spreading_factor', 9),
            'bandwidth': metrics.get('parameters', {}).get('bandwidth', 250)
        }
        
        # Log all available metrics for debugging
        logging.debug(f"[PARAM EVAL] Available metrics: {list(metrics.keys())}")
        logging.debug(f"[PARAM EVAL] Raw metrics: {metrics}")
        
        # Use correct metric keys from aggregator with safe defaults
        snr = metrics.get('avg_ack_snr', metrics.get('avg_packet_snr', 0.0))
        rssi = metrics.get('avg_ack_rssi', metrics.get('avg_packet_rssi', -120.0))
        active_nodes = metrics.get('active_nodes', 1)  # This might not be available
        delivery_rate = metrics.get('delivery_rate', 0.0)
        avg_rtt = metrics.get('avg_rtt_ms', 0.0)
        
        # Ensure all values are numeric for safe formatting
        snr = float(snr) if snr is not None else 0.0
        rssi = float(rssi) if rssi is not None else -120.0
        active_nodes = int(active_nodes) if active_nodes is not None else 1
        delivery_rate = float(delivery_rate) if delivery_rate is not None else 0.0
        avg_rtt = float(avg_rtt) if avg_rtt is not None else 0.0
        
        # Calculate overall success metric for current parameters
        current_params = {
            'spreading_factor': context['parameters']['spreading_factor'],
            'bandwidth': context['parameters']['bandwidth']
        }
        current_success = self._calculate_reward(metrics, current_params)
        
        logging.info(f"[PARAM EVAL] Context: SNR={snr:.1f}, RSSI={rssi:.1f}, Nodes={active_nodes}, Delivery={delivery_rate:.2f}, RTT={avg_rtt:.0f}ms, SF={context['parameters']['spreading_factor']}, BW={context['parameters']['bandwidth']}")
        
        predictions = self.bandit.get_arm_predictions(context)
        
        # Display predictions in a formatted box
        logging.info(f"[PARAM EVAL] Generated {len(predictions)} predictions:")
        predictions_box = self._format_predictions_box(predictions, top_n=18)
        for line in predictions_box.split('\n'):
            logging.info(f"[PARAM EVAL] {line}")
        
        return predictions

    def broadcast_model_predictions(self, predictions, node_id):
        """Broadcast this node's predictions as a compact, ordered byte array.
        
        Args:
            predictions: Dictionary mapping parameter tuples to scores
            node_id: Node identifier for the broadcast message
        """
        import logging
        import json
        # Sort parameter combinations for a fixed order
        param_list = sorted(predictions.keys())
        scores = [predictions[k] for k in param_list]
        # Quantize each score to 0-255 (float -> int, no clamping)
        # Scale to use full range: assume max reasonable value is 10.0 for UCB scores
        max_expected = 10.0
        quantized = [max(0, min(255, int(round(s * 255 / max_expected)))) for s in scores]
        # Compose payload: b'PRED' + node_id (utf-8, 8 bytes, padded/truncated) + quantized bytes
        node_id_bytes = str(node_id).encode('utf-8')[:8].ljust(8, b'_')
        payload = b'PRED' + node_id_bytes + bytes(quantized)
        
        logging.debug(f"[PARAM BROADCAST] Composed payload: {len(payload)} bytes")
        logging.debug(f"[PARAM BROADCAST] Node ID bytes: {node_id_bytes}")
        logging.debug(f"[PARAM BROADCAST] Quantized scores: {quantized[:5]}... (showing first 5)")
        
        if len(payload) > 200:
            logging.error(f"[PARAM BROADCAST] Prediction payload too large ({len(payload)} bytes), not sent!")
            return
        if self.mesh.serial_interface:
            try:
                # Try binary format first
                self.mesh.serial_interface.sendData(payload)
                # Only log the first broadcast, not retransmissions
                if not hasattr(self, '_broadcast_logged'):
                    logging.info(f"[PARAM BROADCAST] Broadcasting {len(scores)} predictions ({len(payload)} bytes)")
                    self._broadcast_logged = True
                
                # Also try text format as backup (in case binary doesn't work)
                text_payload = f"PRED_{json.dumps({'type': 'PARAM_PREDICTIONS', 'node_id': str(node_id), 'scores': scores})}"
                if len(text_payload) <= 200:  # Check size limit
                    self.mesh.serial_interface.sendData(text_payload.encode('utf-8'))
                    logging.debug(f"[PARAM BROADCAST] Also sent text format backup: {len(text_payload)} chars")
                
            except Exception as e:
                logging.error(f"[PARAM BROADCAST] Failed to send prediction message: {e}")
        else:
            logging.error("[PARAM BROADCAST] No serial interface available for broadcasting predictions")
        # Store the order for reconstruction
        self.last_broadcast_param_order = param_list

    def decide_consensus_params(self):
        """Aggregate all received predictions and select the parameter set with highest average expected success.
        
        Returns:
            Dictionary containing the best spreading factor and bandwidth
        """
        import logging
        if not hasattr(self, 'predictions_received') or not self.predictions_received:
            logging.info("[PARAM CONSENSUS] No predictions received, using default params (SF=8, BW=125).")
            return {'spread_factor': 8, 'bandwidth': 125}
        
        # Log participation info and aggregate scores
        node_count = len(self.predictions_received)
        node_ids = list(self.predictions_received.keys())
        logging.info(f"[PARAM CONSENSUS] {node_count} node(s) participated: {node_ids}")
        
        # Aggregate scores for each param set
        from collections import defaultdict
        scores = defaultdict(list)
        for node_id, preds in self.predictions_received.items():
            for k, v in preds.items():
                # k is a tuple (sf, bw)
                try:
                    if isinstance(k, str):
                        k_tuple = eval(k)
                    else:
                        k_tuple = k
                except Exception:
                    continue
                scores[k_tuple].append(v)
        # Compute average
        best_params = None
        best_score = float('-inf')
        for k, vlist in scores.items():
            avg = sum(vlist) / len(vlist)
            if avg > best_score:
                best_score = avg
                best_params = k
        if best_params:
            logging.info(f"[PARAM CONSENSUS] Consensus: SF={best_params[0]}, BW={best_params[1]} (score: {best_score:.3f})")
            return {'spread_factor': best_params[0], 'bandwidth': best_params[1]}
        logging.info("[PARAM CONSENSUS] No valid consensus, using default params (SF=8, BW=125).")
        return {'spread_factor': 8, 'bandwidth': 125}

    def _initialize_minimal_model(self):
        """Initialize the bandit with minimal priors to avoid cold start problems.
        
        This provides very basic initialization without synthetic data that could override real experience.
        """
        logging.info("[PARAM] Initializing minimal model without synthetic data...")
        
        # The bandit will learn from real experience starting with uniform priors
        # No synthetic training data to avoid overriding real experience
        logging.info("[PARAM] Model initialized with minimal priors - will learn from real experience")

# Global variable to hold the current parameter manager instance
param_mgr = None
