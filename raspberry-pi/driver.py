import serial
import time
import os

COMMANDS = {
    "top": (0, "rotate:z,cw"),
    "top_rev": (0, "rotate:z,ccw"),
    "right": (0, "rotate:y,cw"),
    "right_rev": (0, "rotate:y,ccw"),
    "front": (0, "rotate:x,cw"),
    "front_rev": (0, "rotate:x,ccw"),
    "bottom": (1, "rotate:z,cw"),
    "bottom_rev": (1, "rotate:z,ccw"),
    "left": (1, "rotate:y,cw"),
    "left_rev": (1, "rotate:y,ccw"),
    "back": (1, "rotate:x,cw"),
    "back_rev": (1, "rotate:x,ccw")
    }

PORTS = [a for a in os.listdir("/dev") if 'USB' in a]
SLEEP_TIME = 2
SMALL_SLEEP_TIME = 0.01

class Device():
    def __init__(self, port):
        print('Initializing serial...', end = ' ')
        self.serial = serial.Serial(f'/dev/{port}', 9600, timeout=1)
        print('Serial initialized.')
        time.sleep(SLEEP_TIME)
        print("Getting device no...")
        self.device_no = self._get_device()

    def send_command(self, command):
        self.serial.write((command + '\n').encode())
        print(f"Command sent: {command}", end=' | ')
        ack_received = False
        timeout = 2
        start = time.time()
        while not ack_received:
            if self.serial.in_waiting > 0:
                response = self.read_line()
                if response == command + ":ACK":
                    ack_received = True
                    print(f"Arduino response: {response}")
            if time.time() - start > timeout:
                self.send_command(command)
                break
        time.sleep(SMALL_SLEEP_TIME)

    def read_line(self):
        return self.serial.readline().decode('utf-8').rstrip()

    def _get_device(self):
        self.serial.flush()
        self.serial.write(('get_device' + '\n').encode())
        response = self.read_line()
        device_no = int(response)
        print(f"Device {device_no} recognized for port {self.serial.port}")
        return device_no

class Driver():
    def __init__(self):
        self.devices = {}
        for port in PORTS:
            device = Device(port)
            self.devices[device.device_no] = device

    def move_cube(self, move : str):
        assert type(move) == str, f"move must be a string belonging to the set {list(COMMANDS.keys())}"
        assert move in COMMANDS, f"move must belong to the set of moves {list(COMMANDS.keys())}"
        device, command = COMMANDS[move]
        command += ',50'
        self.devices[device].send_command(command)

    def fine_move(self, move : str, steps : int=1):
        device, command = COMMANDS[move]
        command += ',' + str(steps)
        self.devices[device].send_command(command)

    def activate(self):
        for device in self.devices.values():
            device.send_command('activate')

    def deactivate(self):
        for device in self.devices.values():
            device.send_command('deactivate')

    def set_delay(self, delay):
        for device in self.devices.values():
            device.send_command('set_delay:' + str(delay))

if __name__ == "__main__":
    driver = Driver()
    print(driver.devices)