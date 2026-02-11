import tarfile
import struct
import xml.etree.ElementTree as ET
import csv
import sys
import os


def otb_to_csv(otb_path, csv_path=None):
    if csv_path is None:
        csv_path = os.path.splitext(otb_path)[0] + ".csv"

    with tarfile.open(otb_path, "r") as tar:
        # Find the .xml and .sig files
        xml_name = None
        sig_name = None
        for name in tar.getnames():
            if name.endswith(".xml") and not name.startswith("markers") and name not in (
                "DockPanel.config", "form_dock00.xml", "patient.xml"
            ):
                xml_name = name
            if name.endswith(".sig"):
                sig_name = name

        if not xml_name or not sig_name:
            print("Error: Could not find .xml or .sig file inside the archive.")
            sys.exit(1)

        # Parse XML metadata
        xml_data = tar.extractfile(xml_name).read()
        root = ET.fromstring(xml_data)

        num_channels = int(root.attrib["DeviceTotalChannels"])
        sample_freq = int(root.attrib["SampleFrequency"])
        ad_bits = int(root.attrib["ad_bits"])

        # Build channel names from XML
        channel_names = [""] * num_channels
        for adapter in root.iter("Adapter"):
            adapter_desc = adapter.attrib.get("Description", "")
            start_idx = int(adapter.attrib.get("ChannelStartIndex", 0))
            for ch in adapter.iter("Channel"):
                ch_idx = start_idx + int(ch.attrib.get("Index", 0))
                prefix = ch.attrib.get("Prefix", "").strip()
                desc = ch.attrib.get("Description", "").strip()
                if ch_idx < num_channels:
                    channel_names[ch_idx] = f"{prefix}{desc}".strip() or f"Channel_{ch_idx}"

        # Fill any unnamed channels
        for i in range(num_channels):
            if not channel_names[i]:
                channel_names[i] = f"Channel_{i}"

        # Read binary signal data
        sig_data = tar.extractfile(sig_name).read()

    bytes_per_sample = 2  # int16
    total_samples = len(sig_data) // (num_channels * bytes_per_sample)

    print(f"Device: {root.attrib.get('Name', 'Unknown')}")
    print(f"Channels: {num_channels}")
    print(f"Sample frequency: {sample_freq} Hz")
    print(f"AD bits: {ad_bits}")
    print(f"Total samples per channel: {total_samples}")
    print(f"Duration: {total_samples / sample_freq:.2f} seconds")
    print(f"Channel names: {channel_names}")
    print(f"Writing to: {csv_path}")

    # Unpack all data at once for performance
    fmt = f"<{num_channels * total_samples}h"
    all_values = struct.unpack(fmt, sig_data[: num_channels * total_samples * bytes_per_sample])

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Sample", "Time_s"] + channel_names)

        for i in range(total_samples):
            time_s = i / sample_freq
            row = [i, f"{time_s:.6f}"]
            offset = i * num_channels
            for ch in range(num_channels):
                row.append(all_values[offset + ch])
            writer.writerow(row)

    print("Done.")


if __name__ == "__main__":
    otb_file = r"C:\Users\chule\Downloads\0\exp1.otb+"
    otb_to_csv(otb_file)
