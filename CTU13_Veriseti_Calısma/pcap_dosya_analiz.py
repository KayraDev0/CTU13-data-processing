import pyshark

# PCAP dosyasını yükle
cap = pyshark.FileCapture('botnet-capture-20110816-donbot.pcap')

# İlk 10 paket üzerinde temel analiz yapalım
counter = 0
for packet in cap:
    if counter >= 10:
        break
    print(f"Packet #{packet.number}:")
    print(f"Timestamp: {packet.sniff_time}")
    print(f"Packet Length: {packet.length} bytes")
    print(f"Packet Protocol: {packet.transport_layer}")
    if 'IP' in packet:
        print(f"Source IP: {packet.ip.src}")
        print(f"Destination IP: {packet.ip.dst}")
    print('-' * 50)
    counter += 1
