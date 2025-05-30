import meshtastic
import meshtastic.serial_interface
from pubsub import pub
import argparse
import threading
import time
import sys
from datetime import datetime
try:
    from termcolor import colored
except ImportError:
    def colored(text, color=None, attrs=None):
        return text
from pprint import pprint

# Desired LoRa parameters
LORA_PARAMS = {
    'use_preset': False,
    'spread_factor': 7,
    'bandwidth': 125,
    'coding_rate': 8,
    'tx_power': 27
}

# Helper to check and set LoRa parameters and disable telemetry
def ensure_lora_params(node):
    changed = False
    lora = node.localConfig.lora
    # Debug print for each parameter
    print(f"[DEBUG] use_preset: current={lora.use_preset}, desired={LORA_PARAMS['use_preset']}")
    if lora.use_preset != LORA_PARAMS['use_preset']:
        lora.use_preset = LORA_PARAMS['use_preset']
        changed = True
    print(f"[DEBUG] spread_factor: current={lora.spread_factor}, desired={LORA_PARAMS['spread_factor']}")
    if lora.spread_factor != LORA_PARAMS['spread_factor']:
        lora.spread_factor = LORA_PARAMS['spread_factor']
        changed = True
    print(f"[DEBUG] bandwidth: current={lora.bandwidth}, desired={LORA_PARAMS['bandwidth']}")
    if lora.bandwidth != LORA_PARAMS['bandwidth']:
        lora.bandwidth = LORA_PARAMS['bandwidth']
        changed = True
    print(f"[DEBUG] coding_rate: current={lora.coding_rate}, desired={LORA_PARAMS['coding_rate']}")
    if lora.coding_rate != LORA_PARAMS['coding_rate']:
        lora.coding_rate = LORA_PARAMS['coding_rate']
        changed = True
    print(f"[DEBUG] tx_power: current={lora.tx_power}, desired={LORA_PARAMS['tx_power']}")
    if lora.tx_power != LORA_PARAMS['tx_power']:
        lora.tx_power = LORA_PARAMS['tx_power']
        changed = True
    # Disable telemetry environment measurement
    telemetry = getattr(node.localConfig, 'telemetry', None)
    if telemetry is not None:
        print(f"[DEBUG] telemetry.environment_measurement_enabled: current={getattr(telemetry, 'environment_measurement_enabled', None)}, desired=False")
        if hasattr(telemetry, 'environment_measurement_enabled') and telemetry.environment_measurement_enabled:
            telemetry.environment_measurement_enabled = False
            changed = True
    else:
        print("[WARN] No telemetry config found on this node. Skipping telemetry disable.")
    if changed:
        print("[DEBUG] lora config before write:", node.localConfig.lora)
        print("[INFO] LoRa/telemetry parameters changed, writing config and rebooting device...")
        node.writeConfig("lora")
        print("[DEBUG] lora config after write:", node.localConfig.lora)
        node.reboot()
        print("[INFO] Device rebooted. Please restart the script after device reconnects.")
        sys.exit(0)
    else:
        print("[INFO] LoRa and telemetry parameters are correct. No reboot needed.")

# Message receiving thread
class ReceiverThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.running = True
    def run(self):
        while self.running:
            time.sleep(0.1)
    def stop(self):
        self.running = False

def format_packet(packet):
    decoded = packet.get('decoded', {})
    portnum = decoded.get('portnum', 'UNKNOWN')
    ts = datetime.now().strftime('%H:%M:%S')
    # Extract SNR and RSSI if available
    snr = packet.get('rxSnr')
    rssi = packet.get('rxRssi')
    signal_info = ''
    if snr is not None or rssi is not None:
        signal_info = f" [SNR: {snr if snr is not None else '?'} dB, RSSI: {rssi if rssi is not None else '?'} dBm]"
    if portnum == 'TEXT_MESSAGE_APP':
        text = decoded.get('text') or decoded.get('payload', b'').decode(errors='ignore')
        sender = packet.get('fromId', packet.get('from'))
        return colored(f"[{ts}] <{sender}> {text}{signal_info}", "green")
    elif portnum == 'TELEMETRY_APP':
        telemetry = decoded.get('telemetry', {})
        metrics = telemetry.get('deviceMetrics', {})
        battery = metrics.get('batteryLevel', '?')
        voltage = metrics.get('voltage', '?')
        uptime = metrics.get('uptimeSeconds', '?')
        sender = packet.get('fromId', packet.get('from'))
        return colored(f"[{ts}] [TELEMETRY] <{sender}> Battery: {battery}%, Voltage: {voltage}V, Uptime: {uptime}s{signal_info}", "cyan")
    elif portnum == 'ALERT_APP':
        alert = decoded.get('payload', b'').decode(errors='ignore')
        return colored(f"[{ts}] [ALERT] {alert}{signal_info}", "red", attrs=["bold"])
    elif portnum == 'PRIVATE_APP':
        data = decoded.get('payload', b'').decode(errors='ignore')
        return colored(f"[{ts}] [PRIVATE] {data}{signal_info}", "magenta")
    elif portnum == 'NODEINFO_APP':
        user = decoded.get('user', {})
        short = user.get('shortName', '?')
        long = user.get('longName', '?')
        hw = user.get('hwModel', '?')
        nodeid = user.get('id', packet.get('fromId', packet.get('from')))
        return colored(f"[{ts}] [NODEINFO] {short} ({long}) [{hw}] joined the mesh. (id: {nodeid}){signal_info}", "blue", attrs=["bold"])
    else:
        return colored(f"[{ts}] [UNKNOWN] {packet}{signal_info}", "yellow")

def onReceive(packet, interface):
    decoded = packet.get('decoded', {})
    portnum = decoded.get('portnum', 'UNKNOWN')
    
    if portnum == 'PRIVATE_APP' and decoded.get('wantResponse'):
        # Send acknowledgment back to sender
        sender_id = packet.get('fromId')
        message_id = packet.get('id')
        
        # Create acknowledgment payload
        ack_payload = f"ACK:{message_id}"
        
        # Send acknowledgment as PRIVATE_APP or TEXT_MESSAGE_APP
        interface.sendData(ack_payload.encode(), destinationId=sender_id)
        # OR
        # interface.sendText(f"ACK:{message_id}", destinationId=sender_id)
        
        print(f"[ACK] Sent acknowledgment for message {message_id} to {sender_id}")
    print(format_packet(packet))

def list_nodes(interface):
    print("Known nodes:")
    for node_id, node in interface.nodes.items():
        print(f"- {node_id}: {node['user']['longName'] if 'user' in node else 'Unknown'}")

def main():
    parser = argparse.ArgumentParser(description="Meshtastic CLI")
    parser.add_argument('command', nargs='*', help="Command to run (send <msg>, nodes, etc)")
    args = parser.parse_args()

    print("[DEBUG] Connecting to Meshtastic device...")
    interface = meshtastic.serial_interface.SerialInterface()
    print("[DEBUG] Connected. Waiting for device to finish booting...")
    time.sleep(3)
    node = interface.localNode
    print("[DEBUG] Current lora config after connect:", node.localConfig.lora)
    ensure_lora_params(node)

    pub.subscribe(onReceive, "meshtastic.receive")

    # One-shot command mode
    if args.command:
        cmd = args.command
        if cmd[0] == 'send' and len(cmd) > 1:
            msg = ' '.join(cmd[1:])
            interface.sendText(msg)
            print(f"[SENT] {msg}")
        elif cmd[0] == 'nodes':
            list_nodes(interface)
        else:
            print("Unknown command.")
        interface.close()
        return

    # Interactive REPL mode
    print("Meshtastic CLI. Type 'help' for commands.")
    receiver = ReceiverThread()
    receiver.start()
    try:
        while True:
            inp = input('> ').strip()
            if inp == 'exit' or inp == 'quit':
                break
            elif inp.startswith('send '):
                msg = inp[5:]
                interface.sendText(msg)
                print(f"[SENT] {msg}")
            elif inp == 'nodes':
                list_nodes(interface)
            elif inp == 'help':
                print("Commands: send <msg>, nodes, exit, help")
            else:
                print("Unknown command. Type 'help' for commands.")
    except (KeyboardInterrupt, EOFError):
        print("\nExiting...")
    finally:
        receiver.stop()
        interface.close()

if __name__ == '__main__':
    main()