"""
Reads FSR sensor readings. Please ensure port is open: sudo chmod a+rw /dev/ttyACM0
"""

import os
import serial

def fsr_thread(buffer, stop_event):
    """
    This function should be run as an mp process, checks fsr readings
    """
    print("FSR pid:", os.getpid())
    ser = serial.Serial('/dev/ttyACM0', 9600)

    while not stop_event.is_set():
        try:
            line = ser.readline()
            line = line.decode('utf-8').rstrip()
            fsr_val = [0., 0.]
        except UnicodeDecodeError:
            # print("UnicodeDecodeError in decoding line:", line)
            continue

        try:
            fsr0, fsr1 = line.split(',')
            fsr0, fsr1 = fsr0.split(':'), fsr1.split(':')
            # if len(fsr0) > 1 and float(fsr0[1]) < 10000:
            if len(fsr0) > 1 and float(fsr0[1]) < 9999999:
                fsr_val[0] = 1.

            # TODO: After FSR sensors are fixed/replaced, reset these lines
            # if len(fsr1) > 1 and float(fsr1[1]) < 10000:
            #     fsr_val[1] = 1.
                fsr_val[1] = 1.
        
        except ValueError:
            # print("Parsing Error:", line)
            continue

        if buffer.full():
            out = buffer.get(block=True)
        buffer.put(fsr_val)


if __name__ == '__main__':
    ser = serial.Serial('/dev/ttyACM0', 9600)
    while True:
        try:
            line = ser.readline()
            line = line.decode('utf-8').rstrip()
        except UnicodeDecodeError:
            # print("UnicodeDecodeError in decoding line:", line)
            continue
        print(line)

        try:
            fsr0, fsr1 = line.split(',')
            fsr0, fsr1 = fsr0.split(':'), fsr1.split(':')
            if len(fsr0) > 1:
                fsr0_val = float(fsr0[1])
                # print(fsr0_val < 10000)
                print(fsr0_val < 9999999)
            if len(fsr1) > 1:
                fsr1_val = float(fsr1[1])
                # print(fsr1_val < 10000)
                print(fsr1_val < 300000)
        
        except ValueError:
            print("Error", line)


