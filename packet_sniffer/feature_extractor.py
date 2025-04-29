from scapy.all import sniff, IP, TCP, UDP, ICMP
import csv
import time

# CSV File Setup
csv_file = open('logs/packets_features.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['timestamp', 'src_ip', 'dst_ip', 'protocol', 'src_port', 'dst_port', 'packet_length', 'flags'])

def extract_features(packet):
    timestamp = time.time()
    src_ip = dst_ip = protocol = src_port = dst_port = flags = None
    packet_length = len(packet)
    print(f"Packet summary: {packet.summary()}")
    print(f"Initial packet length: {packet_length}")
    
    if IP in packet:
        ip_layer = packet[IP]
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst
        protocol_num = ip_layer.proto
        
        if protocol_num == 6 and TCP in packet:  # TCP
            protocol = "TCP"
            tcp_layer = packet[TCP]
            src_port = tcp_layer.sport
            dst_port = tcp_layer.dport
            flags = tcp_layer.flags
        elif protocol_num == 17 and UDP in packet:  # UDP
            protocol = "UDP"
            udp_layer = packet[UDP]
            src_port = udp_layer.sport
            dst_port = udp_layer.dport
        elif protocol_num == 1 and ICMP in packet:  # ICMP
            protocol = "ICMP"
        else:
            protocol = str(protocol_num)  # Other protocols

        # Save extracted features into CSV
        csv_writer.writerow([timestamp, src_ip, dst_ip, protocol, src_port, dst_port, packet_length, flags])

def start_sniffing(interface="en0"):
    print("[*] Starting Packet Feature Extraction...")
    sniff(prn=extract_features, iface=interface, store=False)

if __name__ == "__main__":
    interface = "en0"  # Default interface (Wi-Fi)
    start_sniffing(interface)
