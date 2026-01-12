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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

        # Initialize rerun
        rr.init("L2F_Simulator", spawn=True)
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
        if self.state.position[2] <= -0.05:
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

        # print(f"RPMs: {rpms} dt: {np.mean(self.simulation_dts):.4f} s, action: {action[0].tolist()}")
        return self.state.position, self.state.orientation, self.state.linear_velocity, self.state.angular_velocity, accelerometer, 101325

    async def run(self):
        self.previous_time = time.time()
        await super().run()