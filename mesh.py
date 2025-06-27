class MeshInterface:
    """Interface for managing mesh network operations and device parameters."""
    
    def __init__(self, config, serial_interface=None):
        """Initialize the mesh interface.
        
        Args:
            config: Configuration dictionary
            serial_interface: Serial interface object (optional)
        """
        self.config = config
        # Initialize hardware or API connection here
        self.serial_interface = serial_interface

    def send(self, packet):
        """Send packet via hardware/API.
        
        Args:
            packet: Dictionary containing packet data
        """
        # Send packet via hardware/API
        if self.serial_interface:
            self.serial_interface.sendData(packet['payload'])

    def receive(self):
        """Receive packet from hardware/API.
        
        Returns:
            Received packet data or None if no data available
        """
        # Receive packet from hardware/API
        if self.serial_interface:
            return self.serial_interface.receive()
        return None

    def reboot_node(self, node_id):
        """Reboot node via hardware/API.
        
        Args:
            node_id: Identifier of the node to reboot
        """
        # Reboot node via hardware/API
        pass

    def update_params(self, params):
        """Update device parameters.
        
        Args:
            params: Dictionary containing new parameters
        """
        # Update device parameters
        pass 

    def get_local_node(self):
        """Get the local node object.
        
        Returns:
            Local node object or None if not available
        """
        # Placeholder: return the local node object
        # In real implementation, connect to hardware/API and return node
        return getattr(self.serial_interface, 'localNode', None)

    def get_node_id(self):
        """Get the local node ID.
        
        Returns:
            Node identifier or None if not available
        """
        node = self.get_local_node()
        if node is None:
            return None
        # Try common attribute names for node ID
        if hasattr(node, 'id'):
            return node.id
        if hasattr(node, 'nodeNum'):
            return node.nodeNum
        # Try dict access if node is a dict-like object
        if isinstance(node, dict):
            return node.get('id') or node.get('nodeNum')
        return None

    def apply_new_params(self, new_params):
        """Apply new LoRa parameters to the device.
        
        Args:
            new_params: Dictionary containing new LoRa parameters
        """
        node = self.get_local_node()
        changed = False
        lora = node.localConfig.lora
        if hasattr(lora, 'spread_factor') and lora.spread_factor != new_params.get('spread_factor'):
            lora.spread_factor = new_params['spread_factor']
            changed = True
        if hasattr(lora, 'bandwidth') and lora.bandwidth != new_params.get('bandwidth'):
            lora.bandwidth = new_params['bandwidth']
            changed = True
        if changed:
            print("[INFO] LoRa parameters changed, writing config and rebooting device...")
            lora.use_preset = False
            node.writeConfig("lora")
            node.reboot()
            print("[INFO] Device rebooted. Reconnecting automatically...")
        else:
            print("[INFO] LoRa parameters are correct. No reboot needed.")

    def ensure_lora_params(self):
        """Check local LoRa parameters against config, update and reboot if needed.
        
        Returns:
            True if reboot was triggered, False otherwise
        """
        node = self.get_local_node()
        if node is None:
            print("[ERROR] No local node found for parameter check.")
            return False
        changed = False
        lora = node.localConfig.lora
        desired = self.config.get('lora_params', {})
        param_map = {
            'spreading_factor': 'spread_factor',
            'bandwidth': 'bandwidth',
            'coding_rate': 'coding_rate',
            'use_preset': 'use_preset',
        }
        for cfg_key, dev_attr in param_map.items():
            desired_val = desired.get(cfg_key)
            if desired_val is not None and hasattr(lora, dev_attr):
                current_val = getattr(lora, dev_attr)
                if current_val != desired_val:
                    print(f"[DEBUG] {dev_attr}: current={current_val}, desired={desired_val}")
                    setattr(lora, dev_attr, desired_val)
                    changed = True
        if changed:
            print("[INFO] LoRa parameters changed, writing config and rebooting device...")
            node.writeConfig("lora")
            node.reboot()
            print("[INFO] Device rebooted. Reconnecting automatically...")
            return True
        else:
            print("[INFO] LoRa parameters are correct. No reboot needed.")
            return False

    def set_ambient_lighting(self, red=0, green=255, blue=0, current=10, led_state=1):
        """Set the ambient lighting LED color to indicate status.
        
        Args:
            red: Red component (0-255, default: 0)
            green: Green component (0-255, default: 255)
            blue: Blue component (0-255, default: 0)
            current: LED current (default: 10)
            led_state: LED state (default: 1)
        """
        node = self.get_local_node()
        if node is None:
            print("[INFO] No local node found for ambient lighting. Node ID:", self.get_node_id())
            return
        ambient = getattr(node.localConfig, 'ambient_lighting', None)
        if ambient is not None:
            ambient.led_state = led_state
            ambient.current = current
            ambient.red = red
            ambient.green = green
            ambient.blue = blue
            print(f"[INFO] Setting ambient lighting: R={red} G={green} B={blue} (current={current})")
            node.writeConfig("ambient_lighting")
        else:
            # Try to flash the screen if possible
            screen = getattr(node.localConfig, 'screen', None)
            if screen and hasattr(screen, 'brightness'):  # Try to flash brightness
                orig_brightness = screen.brightness
                try:
                    screen.brightness = 255
                    node.writeConfig("screen")
                    time.sleep(0.2)
                    screen.brightness = orig_brightness
                    node.writeConfig("screen")
                    print("[INFO] Flashed screen brightness to indicate activity.")
                except Exception:
                    print(f"[INFO] Could not flash screen brightness. Node ID: {self.get_node_id()}")
            else:
                # As a last resort, print a clear info message
                print(f"[INFO] No ambient_lighting or screen config found. Node ID: {self.get_node_id()} is active.") 