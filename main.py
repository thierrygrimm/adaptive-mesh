import argparse
import random
import time
import threading
import meshtastic.serial_interface

import config
import sender
import receiver
import mesh
import aggregator
import parameter_manager
import utils


def experiment_loop(serial_interface, log_file, mesh_iface, param_mgr):
    """Main experiment loop: send, evaluate, reboot, repeat."""
    for cycle in range(5):
        print(f"\n=== Experiment Cycle {cycle+1}/5 ===")
        # 1. Send broadcast packets
        sender.send_broadcast_packets(serial_interface, duration=60, interval=5, log_file=log_file, idle_time=10)
        # 2. Evaluation period (placeholder)
        print("[EVAL] Evaluation period placeholder...")
        time.sleep(5)  # Placeholder for evaluation logic
        # 3. Reboot (placeholder)
        print("[REBOOT] Rebooting device placeholder...")
        # mesh_iface.reboot_node(...) or param_mgr.apply_new_params(...) as needed
        time.sleep(5)  # Placeholder for reboot logic
    print("\n[COMPLETE] Experiment cycles finished.")


def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Adaptive Mesh Network Controller')
    parser.add_argument('--config', type=str, help='Path to config file', default='config.json')
    parser.add_argument('--port', type=str, help='USB serial port for Meshtastic device', default=None)
    args = parser.parse_args()

    # --- Configuration ---
    cfg = config.load_config(args.config)
    # Overwrite config values with CLI arguments if provided
    cli_overrides = {}
    if args.port is not None:
        cli_overrides['port'] = args.port
    if cli_overrides:
        config.merge_config(cfg, cli_overrides)
    run_id = random.randint(10000, 99999)
    log_file = f'sent_packets_{run_id}.json'

    # --- Device Setup ---
    if args.port:
        serial_interface = meshtastic.serial_interface.SerialInterface(devPath=args.port)
    else:
        serial_interface = meshtastic.serial_interface.SerialInterface()
    print("[DEBUG] Serial interface created. Waiting for device boot...")
    time.sleep(3)
    print("[DEBUG] Device boot wait complete. Proceeding with initialization.")

    # --- Parameter Check & Apply ---
    mesh_iface = mesh.MeshInterface(cfg, serial_interface)
    rebooted = mesh_iface.ensure_lora_params()
    if rebooted:
        print("[DEBUG] Waiting for device to reboot and reconnect...")
        time.sleep(8)  # Wait for reboot (adjust as needed)
        # Re-initialize serial interface and mesh_iface
        if args.port:
            serial_interface = meshtastic.serial_interface.SerialInterface(devPath=args.port)
        else:
            serial_interface = meshtastic.serial_interface.SerialInterface()
        print("[DEBUG] Serial interface reconnected after reboot. Waiting for device boot...")
        time.sleep(3)
        mesh_iface = mesh.MeshInterface(cfg, serial_interface)
        print("[DEBUG] Device ready after reboot.")

    # --- Module Initialization ---
    sender_mod = sender.PacketGenerator(cfg, serial_interface, log_file)
    receiver_mod = receiver.PacketReceiver(cfg, serial_interface)
    receiver_mod.start_listener()
    aggregator_mod = aggregator.MetricAggregator(cfg)
    param_mgr = parameter_manager.ParameterCoordinator(cfg, mesh_iface)

    # --- Experiment Execution ---
    experiment_thread = threading.Thread(
        target=experiment_loop,
        args=(serial_interface, log_file, mesh_iface, param_mgr),
        daemon=True
    )
    experiment_thread.start()

    # --- Main Loop: Keep Alive for PubSub/Receiving ---
    try:
        while experiment_thread.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user. Exiting...")


if __name__ == '__main__':
    main()
