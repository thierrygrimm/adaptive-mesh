class MeshInterface:
    def __init__(self, config, serial_interface=None):
        self.config = config
        # Initialize hardware or API connection here
        self.serial_interface = serial_interface

    def send(self, packet):
        # Send packet via hardware/API
        if self.serial_interface:
            self.serial_interface.sendData(packet['payload'])

    def receive(self):
        # Receive packet from hardware/API
        if self.serial_interface:
            return self.serial_interface.receive()
        return None

    def reboot_node(self, node_id):
        # Reboot node via hardware/API
        pass

    def update_params(self, params):
        # Update device parameters
        pass

    def get_local_node(self):
        # Placeholder: return the local node object
        # In real implementation, connect to hardware/API and return node
        return getattr(self.serial_interface, 'localNode', None)

    def apply_new_params(self, new_params):
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
            node.writeConfig("lora")
            node.reboot()
            print("[INFO] Device rebooted. Please restart the script after device reconnects.")
        else:
            print("[INFO] LoRa parameters are correct. No reboot needed.")

    def ensure_lora_params(self):
        """
        Check local LoRa parameters against config, update and reboot if needed.
        Returns True if reboot was triggered, False otherwise.
        """
        node = self.get_local_node()
        if node is None:
            print("[ERROR] No local node found for parameter check.")
            return False
        changed = False
        lora = node.localConfig.lora
        desired = self.config.get('lora_params', {})
        # Map config keys to device attribute names
        param_map = {
            'spreading_factor': 'spread_factor',
            'bandwidth': 'bandwidth',
            'coding_rate': 'coding_rate',
        }
        for cfg_key, dev_attr in param_map.items():
            desired_val = desired.get(cfg_key)
            if desired_val is not None and hasattr(lora, dev_attr):
                current_val = getattr(lora, dev_attr)
                print(f"[DEBUG] {dev_attr}: current={current_val}, desired={desired_val}")
                if current_val != desired_val:
                    setattr(lora, dev_attr, desired_val)
                    changed = True
        if changed:
            print("[INFO] LoRa parameters changed, writing config and rebooting device...")
            node.writeConfig("lora")
            node.reboot()
            print("[INFO] Device rebooted. Continuing after reboot...")
            return True
        else:
            print("[INFO] LoRa parameters are correct. No reboot needed.")
            return False 