import numpy as np
import time
import os
import csv

def save_recording(data, patient_name="unknown"):
    os.makedirs("recordings", exist_ok=True)

    timestamp = int(time.time())
    filename = f"recordings/{patient_name}_{timestamp}.csv"

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Signal"])
        
        for value in data:
            writer.writerow([value])

    return filename