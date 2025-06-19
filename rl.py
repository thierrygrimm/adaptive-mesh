import numpy as np
import logging
from collections import defaultdict
from itertools import product
import json

class ContextualBandit:
    """Contextual bandit using LinUCB algorithm with polynomial feature expansion.
    
    Uses a single shared model that includes target parameters in context.
    """
    
    def __init__(self, config):
        """Initialize the contextual bandit.
        
        Args:
            config: Configuration dictionary containing RL hyperparameters
        """
        self.config = config
        self.alpha = config['rl_hyperparameters'].get('alpha', 1.0)
        self.lambda_reg = config['rl_hyperparameters'].get('lambda_reg', 0.1)
        self.poly_degree = config['rl_hyperparameters'].get('poly_degree', 2)
        
        # Generate all possible parameter combinations
        self.arms = self._generate_arms()
        self.num_arms = len(self.arms)
        
        # Context dimension: network conditions (5) + target parameters (2) = 7
        self.context_dim = 7  # snr, rssi, active_nodes, delivery_rate, avg_rtt, target_sf, target_bw
        self.feature_dim = self._get_polynomial_dim()
        
        # Single shared model for all arms with stronger regularization
        # Increase regularization to prevent overfitting
        effective_lambda = max(self.lambda_reg, 1.0)  # Ensure minimum regularization
        self.A = effective_lambda * np.eye(self.feature_dim)
        self.b = np.zeros(self.feature_dim)
        self.theta = np.zeros(self.feature_dim)
        
        # Performance tracking
        self.performance_history = []
        self.context_history = []
        self.action_history = []
        
        logging.info(f"[RL] Initialized contextual bandit with {self.num_arms} arms, {self.feature_dim} features")
        logging.info(f"[RL] Using single shared model with polynomial degree {self.poly_degree}")
    
    def _generate_arms(self):
        """Generate all possible parameter combinations.
        
        Returns:
            List of dictionaries containing spreading factor and bandwidth combinations
        """
        spreading_factors = [7, 8, 9, 10, 11, 12]
        bandwidths = [125, 250, 500]
        
        arms = []
        for sf in spreading_factors:
            for bw in bandwidths:
                arms.append({'spreading_factor': sf, 'bandwidth': bw})
        
        return arms
    
    def _get_polynomial_dim(self):
        """Calculate dimension of polynomial feature expansion.
        
        Returns:
            Integer dimension of the polynomial feature space
        """
        if self.poly_degree == 1:
            return self.context_dim
        elif self.poly_degree == 2:
            # Linear terms + quadratic terms (including self-interactions)
            return self.context_dim + (self.context_dim * (self.context_dim + 1)) // 2
        else:
            # For higher degrees, use a fixed size to avoid complexity
            return 50
    
    def _extract_context(self, metrics, target_params):
        """Extract context features from metrics and target parameters.
        
        Args:
            metrics: Dictionary containing network performance metrics
            target_params: Dictionary containing target LoRa parameters
            
        Returns:
            Normalized context vector that includes target parameters
        """
        # Extract network condition features using correct metric keys with safe defaults
        snr = metrics.get('avg_ack_snr', metrics.get('avg_packet_snr', 0.0))
        rssi = metrics.get('avg_ack_rssi', metrics.get('avg_packet_rssi', -120.0))
        active_nodes = metrics.get('active_nodes', 1)
        delivery_rate = metrics.get('delivery_rate', 1.0)
        avg_rtt = metrics.get('avg_rtt_ms', 1000.0)
        
        # Handle None values safely
        if snr is None:
            snr = 0.0
        if rssi is None:
            rssi = -120.0
        if active_nodes is None:
            active_nodes = 1
        if delivery_rate is None:
            delivery_rate = 1.0
        if avg_rtt is None:
            avg_rtt = 1000.0
        
        # Extract target parameter features
        target_sf = target_params.get('spreading_factor', 9)
        target_bw = target_params.get('bandwidth', 250)
        
        # Normalize features
        context = np.array([
            (snr + 20) / 40,  # Normalize SNR to [0,1] (assume range -20 to 20)
            (rssi + 140) / 60,  # Normalize RSSI to [0,1] (assume range -140 to -80)
            min(active_nodes / 10.0, 1.0),  # Normalize active nodes to [0,1]
            delivery_rate,  # Already in [0,1]
            min(avg_rtt / 5000.0, 1.0),  # Normalize RTT to [0,1]
            (target_sf - 7) / 5.0,  # Normalize target SF to [0,1]
            (target_bw - 125) / 375.0  # Normalize target BW to [0,1]
        ])
        
        return context
    
    def _polynomial_features(self, context):
        """Generate polynomial features from context vector.
        
        Args:
            context: Input context vector
            
        Returns:
            Polynomial feature vector with consistent dimensions
        """
        if self.poly_degree == 1:
            return context
        
        # Start with linear terms
        features = list(context)  # Convert to list of scalars
        
        if self.poly_degree >= 2:
            # Add quadratic terms (including self-interactions)
            for i in range(len(context)):
                for j in range(i, len(context)):
                    features.append(float(context[i] * context[j]))  # Ensure scalar
        
        # Convert to numpy array
        result = np.array(features)
        
        # Ensure the result has the expected dimension
        if len(result) != self.feature_dim:
            logging.warning(f"[RL] Feature dimension mismatch: expected {self.feature_dim}, got {len(result)}. Adjusting.")
            if len(result) > self.feature_dim:
                result = result[:self.feature_dim]
            else:
                # Pad with zeros if too short
                padding = np.zeros(self.feature_dim - len(result))
                result = np.concatenate([result, padding])
        
        return result
    
    def _calculate_reward(self, metrics, action_params):
        """Calculate reward based on performance metrics.
        
        Args:
            metrics: Dictionary containing performance metrics
            action_params: Dictionary containing action parameters
            
        Returns:
            Float reward value based on delivery rate, latency, and energy efficiency
        """
        delivery_rate = metrics.get('delivery_rate', 1.0)
        avg_rtt = metrics.get('avg_rtt_ms', 1000.0)  # Use correct metric key
        energy_efficiency = metrics.get('energy_efficiency', 1.0)
        
        # More sophisticated reward calculation
        # Delivery rate component (most important)
        delivery_reward = delivery_rate ** 2  # Square to emphasize high delivery rates
        
        # Latency reward (lower RTT is better) - more sensitive
        max_rtt = 10000.0  # Increased max RTT
        min_rtt = 100.0    # Minimum expected RTT
        if avg_rtt <= min_rtt:
            latency_reward = 1.0
        elif avg_rtt >= max_rtt:
            latency_reward = 0.0
        else:
            latency_reward = 1.0 - ((avg_rtt - min_rtt) / (max_rtt - min_rtt)) ** 0.5
        
        # Energy efficiency reward (consider spreading factor impact)
        sf = action_params.get('spreading_factor', 9)
        bw = action_params.get('bandwidth', 250)
        
        # Lower SF and higher BW are generally more energy efficient
        sf_efficiency = (13 - sf) / 6.0  # SF7=1.0, SF12=0.17
        bw_efficiency = bw / 500.0  # 125=0.25, 250=0.5, 500=1.0
        energy_reward = (sf_efficiency + bw_efficiency) / 2.0
        
        # Combine rewards with weights
        total_reward = (delivery_reward * 0.6 + 
                       latency_reward * 0.3 + 
                       energy_reward * 0.1)
        
        # Add small random noise to break ties and encourage exploration
        import random
        noise = random.uniform(-0.01, 0.01)
        total_reward += noise
        
        logging.debug(f"[RL] Reward breakdown for SF={sf}, BW={bw}:")
        logging.debug(f"[RL]   Delivery: {delivery_reward:.3f} (weight: 0.6)")
        logging.debug(f"[RL]   Latency: {latency_reward:.3f} (weight: 0.3)")
        logging.debug(f"[RL]   Energy: {energy_reward:.3f} (weight: 0.1)")
        logging.debug(f"[RL]   Noise: {noise:.3f}")
        logging.debug(f"[RL]   Total: {total_reward:.3f}")
        
        return total_reward
    
    def select_action(self, context):
        """Select action using LinUCB algorithm with epsilon-greedy exploration.
        
        Args:
            context: Dictionary containing current network context
            
        Returns:
            Dictionary containing the best parameter combination
        """
        if not context:
            # Fallback to random selection if no context
            logging.info("[RL] No context available, using random selection")
            return np.random.choice(self.arms)
        
        # Epsilon-greedy exploration
        epsilon = self.config['rl_hyperparameters'].get('epsilon', 0.3)
        if np.random.random() < epsilon:
            # Explore: choose random arm
            selected_arm = np.random.choice(self.arms)
            logging.info(f"[RL] Exploration: randomly selected SF={selected_arm['spreading_factor']}, BW={selected_arm['bandwidth']}")
            return selected_arm
        
        # Exploit: use LinUCB
        best_ucb = float('-inf')
        best_arm = None
        ucb_values = []
        
        logging.info(f"[RL] Evaluating {len(self.arms)} arms with LinUCB...")
        
        for arm in self.arms:
            # Create context with target parameters
            arm_context = context.copy()
            arm_context['parameters'] = arm  # Use arm parameters as target
            
            # Extract and expand context features
            context_vector = self._extract_context(arm_context, arm)
            features = self._polynomial_features(context_vector)
            
            # Calculate UCB using shared model
            A_inv = np.linalg.inv(self.A)
            self.theta = A_inv @ self.b
            
            # UCB = theta^T * x + alpha * sqrt(x^T * A^(-1) * x)
            theta_term = self.theta @ features
            uncertainty_term = self.alpha * np.sqrt(features @ A_inv @ features)
            ucb = theta_term + uncertainty_term
            
            ucb_values.append((arm, ucb, theta_term, uncertainty_term))
            
            if ucb > best_ucb:
                best_ucb = ucb
                best_arm = arm
        
        # Log detailed UCB breakdown for top 3 arms
        ucb_values.sort(key=lambda x: x[1], reverse=True)
        logging.info(f"[RL] Top 3 UCB values:")
        for i, (arm, ucb, theta_term, uncertainty) in enumerate(ucb_values[:3]):
            logging.info(f"[RL]   {i+1}. SF={arm['spreading_factor']}, BW={arm['bandwidth']}: UCB={ucb:.3f} (theta={theta_term:.3f}, uncertainty={uncertainty:.3f})")
        
        logging.info(f"[RL] Exploitation: LinUCB selected SF={best_arm['spreading_factor']}, BW={best_arm['bandwidth']} (UCB={best_ucb:.3f})")
        return best_arm
    
    def update_policy(self, reward, action, context):
        """Update the bandit's policy using the observed reward.
        
        Args:
            reward: Observed reward value
            action: Dictionary containing the action taken
            context: Dictionary containing the context when action was taken
        """
        if not context or not action:
            logging.warning("[RL] Cannot update policy: missing context or action")
            return
        
        # Create context with target parameters
        arm_context = context.copy()
        arm_context['parameters'] = action  # Use action parameters as target
        
        # Extract and expand context features
        context_vector = self._extract_context(arm_context, action)
        features = self._polynomial_features(context_vector)
        
        logging.info(f"[RL] Updating policy for SF={action['spreading_factor']}, BW={action['bandwidth']} with reward {reward:.3f}")
        logging.debug(f"[RL] Context: {context}")
        logging.debug(f"[RL] Action: {action}")
        
        # Log model state before update
        logging.debug(f"[RL] Model state before update:")
        logging.debug(f"[RL]   A matrix condition: {np.linalg.cond(self.A):.2e}")
        logging.debug(f"[RL]   b vector norm: {np.linalg.norm(self.b):.6f}")
        logging.debug(f"[RL]   theta vector norm: {np.linalg.norm(self.theta):.6f}")
        
        # Update shared model
        features_outer = np.outer(features, features)
        self.A += features_outer
        self.b += reward * features
        
        # Update theta with robust matrix inversion
        try:
            A_inv = np.linalg.inv(self.A)
            old_theta = self.theta.copy()
            self.theta = A_inv @ self.b
        except np.linalg.LinAlgError:
            # Use pseudo-inverse as fallback for singular or near-singular matrices
            A_inv = np.linalg.pinv(self.A)
            old_theta = self.theta.copy()
            self.theta = A_inv @ self.b
            logging.debug(f"[RL] Used pseudo-inverse in update due to singular matrix")
        except Exception as e:
            logging.warning(f"[RL] Unexpected error in matrix inversion during update: {e}")
            old_theta = self.theta.copy()
            # Keep current theta if inversion fails
        
        # Log theta change and new state
        theta_change = np.linalg.norm(self.theta - old_theta)
        logging.debug(f"[RL] Theta change: {theta_change:.6f}")
        logging.debug(f"[RL] New theta norm: {np.linalg.norm(self.theta):.6f}")
        
        # Store history
        self.performance_history.append(reward)
        self.context_history.append(context)
        self.action_history.append(action)
        
        logging.info(f"[RL] Updated shared model with reward {reward:.3f} (total updates: {len(self.performance_history)})")
    
    def get_arm_predictions(self, context):
        """Get predictions for all arms given the current context.
        
        Args:
            context: Dictionary containing current network context
            
        Returns:
            Dictionary mapping arm tuples to predicted rewards using LinUCB formula
        """
        if not context:
            return {}
        
        # Calculate theta and A_inv once for all predictions with robust matrix inversion
        try:
            A_inv = np.linalg.inv(self.A)
            self.theta = A_inv @ self.b
        except np.linalg.LinAlgError:
            # Use pseudo-inverse as fallback for singular or near-singular matrices
            A_inv = np.linalg.pinv(self.A)
            self.theta = A_inv @ self.b
            logging.debug(f"[RL] Used pseudo-inverse due to singular matrix")
        except Exception as e:
            logging.warning(f"[RL] Unexpected error in matrix inversion: {e}")
            logging.warning(f"[RL] Using zero theta vector for predictions")
            self.theta = np.zeros(self.feature_dim)
            A_inv = np.eye(self.feature_dim)  # Identity matrix as fallback
        
        # Check for potential overfitting or numerical issues
        if len(self.performance_history) > 0:
            recent_rewards = self.performance_history[-min(10, len(self.performance_history)):]
            avg_recent_reward = np.mean(recent_rewards)
            logging.debug(f"[RL] Recent rewards (last {len(recent_rewards)}): avg={avg_recent_reward:.3f}, min={min(recent_rewards):.3f}, max={max(recent_rewards):.3f}")
        
        # Check if theta is very large (potential overfitting)
        theta_norm = np.linalg.norm(self.theta)
        if theta_norm > 100:
            logging.warning(f"[RL] WARNING: Large theta norm ({theta_norm:.2f}) - potential overfitting!")
        
        # Check if A matrix is ill-conditioned
        condition_number = np.linalg.cond(self.A)
        if condition_number > 1e10:
            logging.warning(f"[RL] WARNING: Ill-conditioned A matrix (condition number: {condition_number:.2e}) - numerical instability!")
        
        predictions = {}
        
        for arm in self.arms:
            # Create context with target parameters
            arm_context = context.copy()
            arm_context['parameters'] = arm  # Use arm parameters as target
            
            # Extract and expand context features
            context_vector = self._extract_context(arm_context, arm)
            features = self._polynomial_features(context_vector)
            
            # Calculate LinUCB prediction: theta^T * x + alpha * sqrt(x^T * A^(-1) * x)
            theta_term = self.theta @ features
            uncertainty_term = self.alpha * np.sqrt(features @ A_inv @ features)
            predicted_reward = theta_term + uncertainty_term
            
            arm_tuple = (arm['spreading_factor'], arm['bandwidth'])
            predictions[arm_tuple] = predicted_reward
            
            # Debug: log if prediction is very high
            if predicted_reward > 1.5:
                logging.debug(f"[RL] DEBUG: SF={arm['spreading_factor']}, BW={arm['bandwidth']}: theta={theta_term:.6f}, uncertainty={uncertainty_term:.6f}, total={predicted_reward:.6f}")
        
        # Check if all predictions are the same (potential issue)
        unique_predictions = set(predictions.values())
        if len(unique_predictions) == 1:
            prediction_value = list(unique_predictions)[0]
            logging.warning(f"[RL] WARNING: All predictions are identical ({prediction_value:.3f}) - model may be stuck!")
            
            # If all predictions are very high, this indicates potential overfitting
            if prediction_value > 2.0:
                logging.warning(f"[RL] CRITICAL: All predictions are very high ({prediction_value:.3f}) - potential overfitting detected!")
                logging.warning(f"[RL] This suggests the model may be overconfident or the uncertainty term is too large")
                logging.warning(f"[RL] Consider: 1) Reducing alpha, 2) Increasing lambda_reg, 3) Reducing poly_degree")
        
        return predictions
    
    def save_model(self, filepath):
        """Save the bandit's state to a JSON file.
        
        Args:
            filepath: Path to save the model file
        """
        import json
        import os
        import logging
        
        try:
            # Prepare data for saving
            model_data = {
                'A': self.A.tolist(),
                'b': self.b.tolist(),
                'theta': self.theta.tolist(),
                'performance_history': self.performance_history,
                'context_history': self.context_history,
                'action_history': self.action_history,
                'config': self.config,
                'arms': self.arms,
                'feature_dim': self.feature_dim,
                'context_dim': self.context_dim,
                'poly_degree': self.poly_degree,
                'alpha': self.alpha,
                'lambda_reg': self.lambda_reg
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(model_data, f, indent=2)
            
            logging.info(f"[RL] Model saved to {filepath}")
            
        except Exception as e:
            logging.error(f"[RL] Failed to save model to {filepath}: {e}")
            raise
    
    def load_model(self, filepath):
        """Load the bandit's state from a JSON file.
        
        Args:
            filepath: Path to load the model file from
        """
        import json
        import logging
        
        try:
            with open(filepath, 'r') as f:
                model_data = json.load(f)
            
            # Restore model state
            self.A = np.array(model_data['A'])
            self.b = np.array(model_data['b'])
            self.theta = np.array(model_data['theta'])
            self.performance_history = model_data['performance_history']
            self.context_history = model_data['context_history']
            self.action_history = model_data['action_history']
            
            # Restore configuration
            self.config = model_data['config']
            self.arms = model_data['arms']
            self.feature_dim = model_data['feature_dim']
            self.context_dim = model_data['context_dim']
            self.poly_degree = model_data['poly_degree']
            self.alpha = model_data['alpha']
            self.lambda_reg = model_data['lambda_reg']
            
            logging.info(f"[RL] Model loaded from {filepath}")
            logging.info(f"[RL] Loaded {len(self.performance_history)} performance records")
            
        except Exception as e:
            logging.error(f"[RL] Failed to load model from {filepath}: {e}")
            raise

class RLAgent:
    """Wrapper class for the contextual bandit to provide a simpler interface."""
    
    def __init__(self, config):
        """Initialize the RL agent.
        
        Args:
            config: Configuration dictionary
        """
        self.bandit = ContextualBandit(config)
    
    def select_action(self, context):
        """Select an action using the contextual bandit.
        
        Args:
            context: Current network context
            
        Returns:
            Selected action parameters
        """
        return self.bandit.select_action(context)
    
    def update_policy(self, reward, action, context):
        """Update the policy with observed reward.
        
        Args:
            reward: Observed reward
            action: Action taken
            context: Context when action was taken
        """
        self.bandit.update_policy(reward, action, context) 