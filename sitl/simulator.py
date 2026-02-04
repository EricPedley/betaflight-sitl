import os
from .sim_base import Simulator
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
import rerun as rr
import torch

from .drone_env import QuadcopterEnv, rotate_vector_by_quaternion_conj

motor_action_remapping = [3,0,1,2] # this is necessary because in betaflight's sitl.c the motor commands are reordered before being sent out

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
            render_mode="human",  # drone_env handles rendering
            device="cpu",
            dynamics_randomization_delta=0.0,
            observation_delay_steps=1,
            auto_reset=False  # Simulator handles its own reset logic
        )
        self.env.reset()

        # Store max_rpm for compatibility
        self.max_rpm = self.env._max_rpm

        self.previous_time = None

        # Initialize rerun with web viewer
        # rr.init("Quadcopter_Simulator", spawn=False)
        # server_uri = rr.serve_grpc()

        # Connect the web viewer to the gRPC server and open it in the browser
        # rr.serve_web_viewer(connect_to=server_uri)

        # print(f"Web viewer available at {server_uri}")
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

        # Convert motor input from Betaflight [0,1] to actions [-1,1]
        action = np.array(motor_input)[motor_action_remapping] * 2 - 1
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

        # Build RC channels to send to Betaflight
        # Channels 0-6: joystick input
        # Channels 7-15: state feedback for NN policy
        vx, vy, vz = linear_velocity
        quat_xyzw = np.array([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
        orientation_rot = R.from_quat(quat_xyzw, scalar_first=False)
        rotvec = orientation_rot.as_rotvec()

        # Compute body frame setpoint error: R^T * (target - position)
        # Target is [0, 0, 1] (hover at 1m height)
        target_position = np.array([0.0, 0.0, 1.0])
        position_error_world = target_position - position
        # Rotate to body frame using R^T (inverse rotation)
        body_setpoint_error = orientation_rot.inv().apply(position_error_world)

        # Compute body frame velocity: R^T * world_velocity
        body_velocity = orientation_rot.inv().apply(linear_velocity)

        rescale = lambda v: max(988, min(2012, int((v + 1) * 512 + 988)))  # PWM: [-1, 1] -> [988, 2012], matches real hardware
        channels = [*self.joystick_values, *([0]*8)]
        # Channels 7-9: body frame setpoint error (direct NN input)
        channels[7] = rescale(body_setpoint_error[0])
        channels[8] = rescale(body_setpoint_error[1])
        channels[9] = rescale(body_setpoint_error[2])
        # Channels 10-12: body frame velocity (direct NN input)
        channels[10] = rescale(body_velocity[0])
        channels[11] = rescale(body_velocity[1])
        channels[12] = rescale(body_velocity[2])
        # Channels 13-15: rotation vector (for quaternion reconstruction)
        channels[13] = rescale(rotvec[0])
        channels[14] = rescale(rotvec[1])
        channels[15] = rescale(rotvec[2])

        # Log RC channels with semantic names
        rr.log("rc/joystick/roll", rr.Scalars(float(self.joystick_values[0])))
        rr.log("rc/joystick/pitch", rr.Scalars(float(self.joystick_values[1])))
        rr.log("rc/joystick/throttle", rr.Scalars(float(self.joystick_values[2])))
        rr.log("rc/joystick/yaw", rr.Scalars(float(self.joystick_values[3])))
        rr.log("rc/joystick/arm", rr.Scalars(float(self.joystick_values[4])))
        rr.log("rc/joystick/mode", rr.Scalars(float(self.joystick_values[5])))
        rr.log("rc/joystick/aux3", rr.Scalars(float(self.joystick_values[6])))
        rr.log("rc/joystick/aux4", rr.Scalars(float(self.joystick_values[7])))

        rr.log("rc/state/body_setpoint_error_x", rr.Scalars(float(body_setpoint_error[0])))
        rr.log("rc/state/body_setpoint_error_y", rr.Scalars(float(body_setpoint_error[1])))
        rr.log("rc/state/body_setpoint_error_z", rr.Scalars(float(body_setpoint_error[2])))
        rr.log("rc/state/body_velocity_x", rr.Scalars(float(body_velocity[0])))
        rr.log("rc/state/body_velocity_y", rr.Scalars(float(body_velocity[1])))
        rr.log("rc/state/body_velocity_z", rr.Scalars(float(body_velocity[2])))
        rr.log("rc/state/rotation_x", rr.Scalars(float(channels[13])))
        rr.log("rc/state/rotation_y", rr.Scalars(float(channels[14])))
        rr.log("rc/state/rotation_z", rr.Scalars(float(channels[15])))

        self.set_rc_channels(channels)

        return position, quaternion, linear_velocity, angular_velocity, accelerometer, 101325, rotor_speeds

    async def run(self):
        self.previous_time = time.time()
        await super().run()
