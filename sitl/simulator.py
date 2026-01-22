import l2f
from betaflight import Simulator
import asyncio
import numpy as np
import json
import time
from scipy.spatial.transform import Rotation as R
import rerun as rr
from .logging_utils import log_drone_pose, log_velocity
import torch
import torch.nn as nn


crazyflie_from_betaflight_motors = [0, 3, 1, 2]

class L2F(Simulator):

    def __init__(self,
            PORT_PWM = 9002,    # Receive RPMs (from Betaflight)
            PORT_STATE = 9003,  # Send state (to Betaflight)
            PORT_RC = 9004,     # Send RC input (to Betaflight)
            UDP_IP = "127.0.0.1",
            SIMULATOR_MAX_RC_CHANNELS=16, # https://github.com/betaflight/betaflight/blob/a94083e77d6258bbf9b8b5388a82af9498c923e9/src/platform/SIMULATOR/target/SITL/target.h#L238
            START_SITL=True,
            AUTO_ARM=False
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

        # Load skrl agent policy network
        self.agent_checkpoint_path = "/home/miller/code/isaac_raptor/logs/skrl/quadrotor_ppo/2026-01-22_09-01-07_ppo_torch/checkpoints/best_agent.pt"
        agent_state = torch.load(self.agent_checkpoint_path, map_location="cpu")
        self.torch_device = torch.device("cpu")

        # Build policy network from checkpoint
        self.policy_net = self._build_policy_network(agent_state)
        

    def _build_policy_network(self, agent_state):
        """Build and load the policy network from checkpoint state."""
        class PolicyNetwork(nn.Module):
            def __init__(self, input_size=12, hidden_size=32, output_size=4):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ELU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ELU()
                )
                self.mean = nn.Linear(hidden_size, output_size)
                self.log_std = nn.Parameter(torch.zeros(output_size))

            def forward(self, x):
                features = self.net(x)
                return self.mean(features)

        policy_net = PolicyNetwork(input_size=12, hidden_size=32, output_size=4)
        policy_net.to(self.torch_device)

        # Load weights from checkpoint
        policy_weights = agent_state["policy"]
        state_dict = {}
        state_dict["net.0.weight"] = policy_weights["net_container.0.weight"]
        state_dict["net.0.bias"] = policy_weights["net_container.0.bias"]
        state_dict["net.2.weight"] = policy_weights["net_container.2.weight"]
        state_dict["net.2.bias"] = policy_weights["net_container.2.bias"]
        state_dict["mean.weight"] = policy_weights["policy_layer.weight"]
        state_dict["mean.bias"] = policy_weights["policy_layer.bias"]
        state_dict["log_std"] = policy_weights["log_std_parameter"]

        policy_net.load_state_dict(state_dict)
        policy_net.eval()
        return policy_net

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


        # Convert quaternion from [w, x, y, z] to [x, y, z, w] for rotation matrix
        quat_xyzw = np.array([self.state.orientation[1], self.state.orientation[2],
                              self.state.orientation[3], self.state.orientation[0]])
        r = R.from_quat(quat_xyzw, scalar_first=False)
        R_wb = r.as_matrix()  # World to body rotation matrix

        # Body frame linear velocity
        body_linear_vel = R_wb.T @ self.state.linear_velocity

        # Body frame angular velocity
        body_angular_vel = self.state.angular_velocity

        # Body frame projected gravity (gravity vector [0, 0, -9.81] transformed to body frame)
        world_gravity = np.array([0, 0, -1], dtype=np.float64)
        body_gravity = R_wb.T @ world_gravity

        # Body frame position setpoint (0, 0, 1) for now
        body_position_setpoint = R_wb.T @ (np.array([0, 0, 1], dtype=np.float32) - self.state.position)

        # Concatenate into 12D observation vector
        obs_array = np.concatenate([
            body_linear_vel,
            body_angular_vel,
            body_gravity,
            body_position_setpoint
        ]).astype(np.float32)

        observation = torch.from_numpy(obs_array).unsqueeze(0).to(self.torch_device)

        # Compute action using policy network
        with torch.no_grad():
            action_tensor = self.policy_net(observation)
            action = action_tensor.squeeze(0).cpu().numpy()

        action = np.clip(action, -1, 1)
        dts = l2f.step(self.device, self.env, self.params, self.state, action, self.next_state, self.rng)
        acceleration = (self.next_state.linear_velocity - self.state.linear_velocity) / simulation_dt
        r = R.from_quat([*self.state.orientation[1:], self.state.orientation[0]])
        R_wb = r.as_matrix()
        accelerometer = R_wb.T @ (acceleration - np.array([0, 0, -9.81], dtype=np.float64))
        self.state.assign(self.next_state)
        if self.state.position[2] <= -1.001:
            self.state.position[2] = -1
            self.state.linear_velocity[:] = 0
            self.state.orientation = [1, 0, 0, 0]
            self.state.angular_velocity[:] = 0

        # Log to rerun instead of sending to websocket
        # Convert quaternion from [w, x, y, z] to [x, y, z, w] for rerun
        quat_xyzw = np.array([self.state.orientation[1], self.state.orientation[2],
                              self.state.orientation[3], self.state.orientation[0]])

        # Apply rotation matrix to position and velocity

        orientation_rot = R.from_quat(quat_xyzw, scalar_first=False)

        log_drone_pose(position=self.state.position, quaternion=quat_xyzw)
        log_velocity(position=self.state.position, velocity=self.state.linear_velocity)
        # rc channels to send should be x,y,yaw,z,aux, vx, vy, vz
        x,y,z = self.state.position
        vx, vy, vz = self.state.linear_velocity
        rescale = lambda v: int((v + 1) * 500 + 1000)
        channels = [*self.joystick_values,*([0]*8)]
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
        rr.log("rpms", rr.Scalars(rpms))

        self.set_rc_channels(channels)

        # print(f"RPMs: {rpms} dt: {np.mean(self.simulation_dts):.4f} s, action: {action[0].tolist()}")
        return self.state.position, self.state.orientation, self.state.linear_velocity, self.state.angular_velocity, accelerometer, 101325

    async def run(self):
        self.previous_time = time.time()
        await super().run()