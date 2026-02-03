import os
from .sim_base import Simulator
import asyncio
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
import rerun as rr
from .logging_utils import log_drone_pose, log_velocity
import torch
import torch.nn as nn

from .drone_env import QuadcopterEnv, rotate_vector_by_quaternion_conj



class L2F(Simulator):

    def __init__(self,
            PORT_PWM = 9002,    # Receive RPMs (from Betaflight)
            PORT_STATE = 9003,  # Send state (to Betaflight)
            PORT_RC = 9004,     # Send RC input (to Betaflight)
            UDP_IP = "127.0.0.1",
            SIMULATOR_MAX_RC_CHANNELS=16, # https://github.com/betaflight/betaflight/blob/a94083e77d6258bbf9b8b5388a82af9498c923e9/src/platform/SIMULATOR/target/SITL/target.h#L238
            AUTO_ARM=False,
            parameters_file=None
        ):
        super().__init__(PORT_PWM, PORT_STATE, PORT_RC, UDP_IP, SIMULATOR_MAX_RC_CHANNELS)

        # Find parameters file
        if parameters_file:
            config_path = parameters_file
        else:
            raise FileNotFoundError(
                f"No parameters file provided"
                "Please provide parameters_file argument."
            )

        # Initialize physics engine from drone_env
        self.env = QuadcopterEnv(
            num_envs=1,
            config_path=config_path,
            dt=0.001,  # Will be updated dynamically
            render_mode=None,  # We handle rendering ourselves
            device="cpu",
            auto_reset=False  # Simulator handles its own reset logic
        )
        self.env.reset()

        # Store max_rpm for compatibility
        self.max_rpm = self.env._max_rpm

        self.previous_time = None

        # Initialize rerun with web viewer
        rr.init("Quadcopter_Simulator", spawn=False)
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

    async def step(self, motor_input):
        simulation_dt = time.time() - self.previous_time
        self.previous_time = time.time()

        # Update env dt
        self.env.dt = simulation_dt

        # Store previous velocity for accelerometer calculation
        prev_velocity = self.env._velocity[0].clone()

        # Convert quaternion from [w, x, y, z] to [x, y, z, w] for rotation matrix
        quat_wxyz = self.env._quaternion[0].cpu().numpy()
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        r = R.from_quat(quat_xyzw, scalar_first=False)
        R_wb = r.as_matrix()  # World to body rotation matrix

        # Body frame linear velocity
        body_linear_vel = R_wb.T @ self.env._velocity[0].cpu().numpy()

        # Body frame angular velocity
        body_angular_vel = self.env._angular_velocity[0].cpu().numpy()

        # Body frame projected gravity (gravity vector [0, 0, -1] transformed to body frame)
        world_gravity = np.array([0, 0, -1], dtype=np.float64)
        body_gravity = R_wb.T @ world_gravity

        # Body frame position setpoint (0, 0, 1) for now
        body_position_setpoint = R_wb.T @ (np.array([0, 0, 1], dtype=np.float32) - self.env._position[0].cpu().numpy())

        # Concatenate into 12D observation vector
        obs_array = np.concatenate([
            body_linear_vel,
            body_angular_vel,
            body_gravity,
            body_position_setpoint
        ]).astype(np.float32)

        # Override with motor input from Betaflight (convert [0,1] to [-1,1] with motor remapping)
        action = np.array(motor_input) * 2 - 1
        actions_tensor = torch.tensor(action, dtype=torch.float32, device=self.env.device).unsqueeze(0)

        # Step physics (bypass decimation, step once with current dt)
        self.env._actions = actions_tensor.clone().clamp(-1.0, 1.0)
        actions_0_1 = self.env._max_rpm * (self.env._actions + 1.0) / 2.0
        self.env._step_once(actions_0_1)

        # Compute accelerometer (specific force in body frame)
        velocity_change = self.env._velocity[0] - prev_velocity
        acceleration_world = velocity_change / max(simulation_dt, 1e-6)
        gravity_world = torch.tensor([0.0, 0.0, -9.81], device=self.env.device)
        specific_force_world = acceleration_world - gravity_world
        # Transform to body frame
        specific_force_body = rotate_vector_by_quaternion_conj(
            specific_force_world.unsqueeze(0),
            self.env._quaternion
        )[0]
        accelerometer = specific_force_body.cpu().numpy()

        # Extract state
        position = self.env._position[0].cpu().numpy()
        quaternion = self.env._quaternion[0].cpu().numpy()  # (w, x, y, z)
        linear_velocity = self.env._velocity[0].cpu().numpy()
        angular_velocity = self.env._angular_velocity[0].cpu().numpy()
        rotor_speeds = self.env._rotor_speeds[0].cpu().numpy()

        # Ground collision handling
        if position[2] <= 0.0:
            position[2] = 0.0
            self.env._position[0, 2] = 0.0
            self.env._velocity[0] = 0.0
            self.env._quaternion[0] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.env.device)
            self.env._angular_velocity[0] = 0.0
            linear_velocity = np.zeros(3)
            angular_velocity = np.zeros(3)
            quaternion = np.array([1.0, 0.0, 0.0, 0.0])

        # Log to rerun
        # Convert quaternion from [w, x, y, z] to [x, y, z, w] for rerun
        quat_xyzw = np.array([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
        orientation_rot = R.from_quat(quat_xyzw, scalar_first=False)

        log_drone_pose(position=position, quaternion=quat_xyzw)
        log_velocity(position=position, velocity=linear_velocity)

        # rc channels to send should be x,y,yaw,z,aux, vx, vy, vz
        x, y, z = position
        vx, vy, vz = linear_velocity
        rescale = lambda v: int((v + 1) * 500 + 1000) % 32768
        channels = [*self.joystick_values, *([0]*8)]
        channels[7] = rescale(x)
        channels[8] = rescale(y)
        channels[9] = rescale(z)
        channels[10] = rescale(vx)
        channels[11] = rescale(vy)
        channels[12] = rescale(vz)
        channels[13] = rescale(orientation_rot.as_rotvec()[0])
        channels[14] = rescale(orientation_rot.as_rotvec()[1])
        channels[15] = rescale(orientation_rot.as_rotvec()[2])

        # Log RC channels to rerun as scalars
        rr.log("rc_channels", rr.Scalars(channels))
        rr.log("motor_command", rr.Scalars(motor_input))
        rr.log("actions", rr.Scalars(action))

        self.set_rc_channels(channels)

        return position, quaternion, linear_velocity, angular_velocity, accelerometer, 101325, rotor_speeds

    async def run(self):
        self.previous_time = time.time()
        await super().run()
