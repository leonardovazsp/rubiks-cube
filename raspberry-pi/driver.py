import serial
import time

COMMANDS = {
    "top": (0, "zcw"),
    "top_rev": (0, "zccw"),
    "right": (0, "ycw"),
    "right_rev": (0, "yccw"),
    "front": (0, "xcw"),
    "front_rev": (0, "xccw"),
    "bottom": (1, "zcw"),
    "bottom_rev": (1, "zccw"),
    "left": (1, "ycw"),
    "left_rev": (1, "yccw"),
    "back": (1, "xcw"),
    "back_rev": (1, "xccw")
    }

PORTS = ['ttyUSB0', 'ttyUSB0']
SLEEP_TIME = 2

class Device():
    def __init__(self, port):
        self.serial = serial.Serial(f'/dev/{port}', 9600, timeout=1)
        time.sleep(SLEEP_TIME)
        self.device_no = self._get_device()

    def send_command(self, command):
        self.serial.write((command + '\n').encode())

    def read_line(self):
        return self.serial.readline().decode('utf-8').rstrip()

    def _get_device(self):
        self.serial.flush()
        self.send_command('get_device')
        device_no = int(self.read_line())
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
        self.devices[device].send_command(command)