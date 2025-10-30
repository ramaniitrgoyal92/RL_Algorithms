import jax
from jax import numpy as jnp
from flax import struct
from functools import partial

from .multibase import MultiBase, State


@struct.dataclass
class State:
    pipeline_state: jnp.ndarray
    reward: jnp.ndarray
    mask: jnp.ndarray
    collision: jnp.ndarray


class Multi2d(MultiBase):

    def __init__(self, num_agents):
        super().__init__(num_agents)
        self.num_agents = num_agents
        self.action_dim_agent = 2
        self.obsv_dim_agent = 4
        self.pos_dim_agent = 2
        self.diameter = 5
        self.agent_radius = 0.15
        self.safe_margin = 0.05
        self.stop_distance = self.agent_radius / 2  # max distance to goal for termination
        self.stop_velocity = float('inf')  # max velocity for termination when reach the goal
        self.mav = jnp.pi / 2  # max angular velocity
        self.mlv = 1.0  # max linear velocity
        self.mla = 1.0  # max linear acceleration

        initial_states, goal_states = self.generate_positions(self.diameter, num_agents)
        self.directions = goal_states[:, :self.pos_dim_agent] - initial_states[:, :self.pos_dim_agent]
        angles = jnp.arctan2(self.directions[:, 1], self.directions[:, 0])
        initial_states = initial_states.at[:, 2].set(angles)

        self.lim = self.diameter / 2 + 1

        self.x0 = initial_states.flatten()
        self.xg = goal_states.flatten()
        self.max_distance = self.diameter

    @partial(jax.jit, static_argnums=(0,))
    def agent_dynamics(self, x, u):
        """
        x[0]: position x
        x[1]: position y
        x[2]: heading angle theta
        x[3]: linear velocity v
        u[0]: angular velocity
        u[1]: linear acceleration

        Returns the time derivative of the state.
        """
        # Constants
        max_angular_velocity, max_linear_velocity, max_linear_acceleration = self.mav, self.mlv, self.mla

        # Extract state variables
        x_pos, y_pos, theta, v = x
        angular_velocity = jnp.clip(u[0], -max_angular_velocity, max_angular_velocity)
        linear_acceleration = jnp.clip(u[1], -max_linear_acceleration, max_linear_acceleration)

        # Dynamics equations
        x_dot = v * jnp.cos(theta)
        y_dot = v * jnp.sin(theta)
        theta_dot = angular_velocity
        v_dot = linear_acceleration

        return jnp.array([x_dot, y_dot, theta_dot, v_dot])

    @partial(jax.jit, static_argnums=(0,))
    def clip_actions(self, traj: jax.Array, factor=1):
        traj = traj.reshape(-1, self.num_agents, self.action_dim_agent)
        traj = jnp.stack([
            jnp.clip(traj[..., 0], -self.mav * factor, self.mav * factor),
            jnp.clip(traj[..., 1], -self.mla * factor, self.mla * factor)
        ], axis=-1)
        return traj.reshape(-1, self.action_dim_agent * self.num_agents)
    
    def clip_velocity(self, x):
        v = jnp.clip(x[3], -self.mlv, self.mlv)
        return x.at[3].set(v)
    
    def get_current_velocity(self, q):
        return q[:, 3]
    
    @property
    def action_size(self):
        return self.action_dim_agent * self.num_agents  # Two actions (steering and acceleration) per agent
    
    def get_heading_line(self, state, position, agent_idx):
        theta = state[2]
        dx = self.agent_radius * jnp.cos(theta)
        dy = self.agent_radius * jnp.sin(theta)
        return [position[0], position[0] + dx], [position[1], position[1] + dy]
