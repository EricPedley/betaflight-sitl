import l2f
from betaflight import Simulator
import asyncio
import numpy as np
import json
import time
from scipy.spatial.transform import Rotation as R
import rerun as rr
from .logging_utils import log_drone_pose, log_velocity


crazyflie_from_betaflight_motors = [0, 3, 1, 2]

class L2F(Simulator):

    def __init__(self,
            PORT_PWM = 9002,    # Receive RPMs (from Betaflight)
            PORT_STATE = 9003,  # Send state (to Betaflight)
            PORT_RC = 9004,     # Send RC input (to Betaflight)
            UDP_IP = "127.0.0.1",
            SIMULATOR_MAX_RC_CHANNELS=16, # https://github.com/betaflight/betaflight/blob/a94083e77d6258bbf9b8b5388a82af9498c923e9/src/platform/SIMULATOR/target/SITL/target.h#L238
            START_SITL=True,
            AUTO_ARM=True
        ):
        super().__init__(PORT_PWM, PORT_STATE, PORT_RC, UDP_IP, SIMULATOR_MAX_RC_CHANNELS, START_SITL)
        self.device = l2f.Device()
        self.rng = l2f.Rng()
        self.env = l2f.Environment()
        self.params = l2f.Parameters()
        self.state = l2f.State()
        self.next_state = l2f.State()
        l2f.initialize_rng(self.device, self.rng, 0)
        l2f.initialize_environment(self.device, self.env)
        l2f.sample_initial_parameters(self.device, self.env, self.params, self.rng)
        l2f.initial_state(self.device, self.env, self.params, self.state)
        self.previous_time = None
        self.simulation_dts = []

        # Initialize rerun with web viewer
        rr.init("L2F_Simulator", spawn=False)
        server_uri = rr.serve_grpc()

        # Connect the web viewer to the gRPC server and open it in the browser
        rr.serve_web_viewer(connect_to=server_uri)

        print(f"Web viewer available at {server_uri}")
        self.joystick_values = [0]*8
        if AUTO_ARM:
            self.joystick_values[4] = 2000
            self.joystick_values[5] = 2000
    
    def set_joystick_channels(self, joystick_values):
        self.joystick_values = joystick_values

    async def step(self, rpms):
        simulation_dt = time.time() - self.previous_time
        self.previous_time = time.time()
        self.simulation_dts.append(simulation_dt)
        self.simulation_dts = self.simulation_dts[-100:]

        parameters_string = l2f.parameters_to_json(self.device, self.env, self.params)
        parameters = json.loads(parameters_string)
        parameters["integration"]["dt"] = simulation_dt
        l2f.parameters_from_json(self.device, self.env, json.dumps(parameters), self.params)

        action = np.array(rpms)[crazyflie_from_betaflight_motors] * 2 - 1
        dts = l2f.step(self.device, self.env, self.params, self.state, action, self.next_state, self.rng)
        acceleration = (self.next_state.linear_velocity - self.state.linear_velocity) / simulation_dt
        r = R.from_quat([*self.state.orientation[1:], self.state.orientation[0]])
        R_wb = r.as_matrix()
        accelerometer = R_wb.T @ (acceleration - np.array([0, 0, -9.81], dtype=np.float64))
        self.state.assign(self.next_state)
        if self.state.position[2] <= -0.001:
            self.state.position[2] = 0
            self.state.linear_velocity[:] = 0
            self.state.orientation = [1, 0, 0, 0]
            self.state.angular_velocity[:] = 0

        # Log to rerun instead of sending to websocket
        # Convert quaternion from [w, x, y, z] to [x, y, z, w] for rerun
        quat_xyzw = np.array([self.state.orientation[1], self.state.orientation[2],
                              self.state.orientation[3], self.state.orientation[0]])

        # NED to ENU rotation matrix
        # NED: x=North, y=East, z=Down -> ENU: x=East, y=North, z=Up
        # This is equivalent to: [0 1 0; 1 0 0; 0 0 -1] matrix
        ned_to_enu_matrix = np.array([[0, 1, 0],
                                       [1, 0, 0],
                                       [0, 0, -1]], dtype=np.float64)

        # Apply rotation matrix to position and velocity
        position_enu = ned_to_enu_matrix @ np.array(self.state.position)
        velocity_enu = ned_to_enu_matrix @ np.array(self.state.linear_velocity)

        # Apply rotation to quaternion
        ned_to_enu_rot = R.from_matrix(ned_to_enu_matrix)
        quat_ned = R.from_quat(quat_xyzw)
        quat_enu = ned_to_enu_rot * quat_ned
        quat_enu_xyzw = quat_enu.as_quat()

        log_drone_pose(position=position_enu, quaternion=quat_enu_xyzw)
        log_velocity(position=position_enu, velocity=velocity_enu)
        # rc channels to send should be x,y,yaw,z,aux, vx, vy, vz
        x,y,z = position_enu
        vx, vy, vz = velocity_enu
        yaw = quat_enu.as_euler('zyx')[0]
        rescale = lambda v: int((v + 1) * 500 + 1000)
        x_int = rescale(x/10)
        y_int = rescale(y/10)
        z_int = rescale(z/10)
        yaw_int = rescale(yaw/np.pi)
        vx_int = rescale(vx/10)
        vy_int = rescale(vy/10)
        vz_int = rescale(vz/10)
        channels = [*self.joystick_values, x_int,y_int,z_int,yaw_int,vx_int,vy_int,vz_int, 0]


        # Log RC channels to rerun as scalars
        rr.log("rc_channels", rr.Scalars(channels))
        rr.log("rpms", rr.Scalars(rpms))

        self.set_rc_channels(channels)

        # print(f"RPMs: {rpms} dt: {np.mean(self.simulation_dts):.4f} s, action: {action[0].tolist()}")
        return self.state.position, self.state.orientation, self.state.linear_velocity, self.state.angular_velocity, accelerometer, 101325

    async def run(self):
        self.previous_time = time.time()
        await super().run()