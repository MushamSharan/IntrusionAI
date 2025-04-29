from scapy.all import sniff
import time

def process_packet(packet):
    print(f"Packet: {packet.summary()}")

def start_sniffing(interface="en0"):
    print("[*] Starting Packet Sniffer...")
    sniff(prn=process_packet, iface=interface, store=False)

if __name__ == "__main__":
    interface = "en0"  # Default Wi-Fi interface on macOS
    start_sniffing(interface)
