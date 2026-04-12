import serial

ser = serial.Serial('COM3', 115200)  # change COM if needed

def read_serial():
    try:
        line = ser.readline().decode().strip()
        if line.isdigit():
            return int(line)
    except:
        return None