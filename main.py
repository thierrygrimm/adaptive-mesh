import argparse
import config
import sender
import receiver
import mesh
import rl_policy
import aggregator
import parameter_manager
import utils
import meshtastic.serial_interface
import random
import time


def main():
    parser = argparse.ArgumentParser(description='Adaptive Mesh Network Controller')
    parser.add_argument('--config', type=str, help='Path to config file', default='config.json')
    parser.add_argument('--port', type=str, help='USB serial port for Meshtastic device', default=None)
    args = parser.parse_args()
    # Load configuration
    cfg = config.load_config(args.config)
    # Generate a random run ID for log file uniqueness
    run_id = random.randint(10000, 99999)
    log_file = f'sent_packets_{run_id}.json'
    # Retrieve serial interface from USB (optionally with port)
    if args.port:
        serial_interface = meshtastic.serial_interface.SerialInterface(devPath=args.port)
    else:
        serial_interface = meshtastic.serial_interface.SerialInterface()
    # Initialize modules with serial interface where needed
    mesh_iface = mesh.MeshInterface(cfg, serial_interface)
    sender_mod = sender.PacketGenerator(cfg, serial_interface, log_file)
    receiver_mod = receiver.PacketReceiver(cfg, serial_interface)
    receiver_mod.start_listener()
    rl_agent = rl_policy.RLAgent(cfg)
    aggregator_mod = aggregator.MetricAggregator(cfg)
    param_mgr = parameter_manager.ParameterCoordinator(cfg, mesh_iface)
    # Main experiment loop: send, evaluate, reboot, repeat
    for cycle in range(5):
        print(f"\n=== Experiment Cycle {cycle+1}/5 ===")
        # 1. Send broadcast packets
        sender.send_broadcast_packets(serial_interface, duration=60, interval=0.1, log_file=log_file, idle_time=10)
        # 2. Evaluation period (placeholder)
        print("[EVAL] Evaluation period placeholder...")
        time.sleep(5)  # Placeholder for evaluation logic
        # 3. Reboot (placeholder)
        print("[REBOOT] Rebooting device placeholder...")
        # mesh_iface.reboot_node(...) or param_mgr.apply_new_params(...) as needed
        time.sleep(5)  # Placeholder for reboot logic
    print("\n[COMPLETE] Experiment cycles finished.")

if __name__ == '__main__':
    main()
