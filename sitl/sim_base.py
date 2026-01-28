
import struct
import time
import numpy as np
import asyncio
import socket
import os
import subprocess


async def udp_recv(loop, sock):
    return await loop.sock_recv(sock, 1024)

def flu_to_frd(v):
    return np.array([v[0], -v[1], -v[2]], dtype=np.float64)

class Simulator():
    def __init__(self,
            PORT_PWM = 9002,    # Receive RPMs (from Betaflight)
            PORT_STATE = 9003,  # Send state (to Betaflight)
            PORT_RC = 9004,     # Send RC input (to Betaflight)
            UDP_IP = "127.0.0.1",
            SIMULATOR_MAX_RC_CHANNELS=16, # https://github.com/betaflight/betaflight/blob/a94083e77d6258bbf9b8b5388a82af9498c923e9/src/platform/SIMULATOR/target/SITL/target.h#L238
            START_SITL=True
        ):
        if START_SITL:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            SITL_EXECUTABLE = "betaflight_4.6.0_SITL"
            subprocess.Popen([os.path.join(current_dir, "firmware", "obj", SITL_EXECUTABLE)], cwd=os.path.dirname(os.path.abspath(__file__)))
            time.sleep(2)
        self.PORT_PWM = PORT_PWM
        self.PORT_STATE = PORT_STATE
        self.PORT_RC = PORT_RC
        self.UDP_IP = UDP_IP
        self.SIMULATOR_MAX_RC_CHANNELS = SIMULATOR_MAX_RC_CHANNELS
        self.udp_pwm_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_pwm_sock.bind((UDP_IP, PORT_PWM))
        self.udp_pwm_sock.setblocking(False)
        self.udp_state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_rc_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rc_channels = [1500] * self.SIMULATOR_MAX_RC_CHANNELS 
    
    def set_rc_channels(self, channels):
        '''
            Array. Value range: 1000-2000
        '''
        self.rc_channels = channels[:self.SIMULATOR_MAX_RC_CHANNELS]
        while len(self.rc_channels) < self.SIMULATOR_MAX_RC_CHANNELS:
            self.rc_channels.append(1000)

    async def step(self, rpms):
        # Placeholder for the step function
        # returns position, orientation, linear_velocity, angular_velocity, accelerometer, pressure # [m, (w, x, y, z) quaternion, m/s, rad/s, m/s^2, ?] all in FLU
        return np.zeros(3), np.array([1, 0, 0, 0]), np.zeros(3), np.zeros(3), np.zeros(3), 101325
    
    async def run(self):
        loop = asyncio.get_event_loop()
        while True:
            try:
                data = await asyncio.wait_for(udp_recv(loop, self.udp_pwm_sock), timeout=0.01)
                rpms = struct.unpack('<4f', data[:16]) if len(data) >= 16 else [0.0, 0.0, 0.0, 0.0]
                rpms = np.clip(rpms, 0, 1.0)
            except asyncio.TimeoutError:
                rpms = [0.0, 0.0, 0.0, 0.0]

            timestamp = time.time()
            position, orientation, linear_velocity, angular_velocity, accelerometer, pressure, rpm_measurement = await self.step(rpms)
            rc_packet = struct.pack(f'<d{self.SIMULATOR_MAX_RC_CHANNELS}h', timestamp, *self.rc_channels)
            self.udp_rc_sock.sendto(rc_packet, (self.UDP_IP, self.PORT_RC))
            packet = struct.pack('<d3d3d4d3d3dd4d',
                timestamp,
                *flu_to_frd(angular_velocity),
                *(-accelerometer),
                *orientation,
                *position,
                *linear_velocity,
                pressure,
                *rpm_measurement
            )
            self.udp_state_sock.sendto(packet, (self.UDP_IP, self.PORT_STATE))
