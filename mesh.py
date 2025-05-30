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