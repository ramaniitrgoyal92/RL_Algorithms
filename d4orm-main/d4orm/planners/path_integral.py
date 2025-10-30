import functools
import jax
from jax import numpy as jnp
from jax import config
from dataclasses import dataclass
import tyro
from tqdm import tqdm
from matplotlib import pyplot as plt
import time
import os

import d4orm

# NOTE: this is important for simulating long horizon open loop control
# config.update("jax_enable_x64", True)


## load config
@dataclass
class Args:
    # exp
    seed: int = 0
    disable_recommended_params: bool = False
    method: str = "mppi"  # mppi, cma-es, cem
    # env
    env_name: str = "multi2dholo"
    Nagent: int = 8  # number of agents
    # diffusion
    Nsample: int = 2048  # number of samples
    Hsample: int = 100  # horizon
    Niteration: int = 100  # number of repeat steps
    temp_sample: float = 0.3  # temperature for sampling
    save_data: bool = False
    save_images: bool = False


@jax.jit
def softmax_update(weights, Y0s, sigma, mu_0t):
    mu_0tm1 = jnp.einsum("n,nij->ij", weights, Y0s)
    return mu_0tm1, sigma


@jax.jit
def cma_es_update(weights, Y0s, sigma, mu_0t):
    mu_0tm1 = jnp.einsum("n,nij->ij", weights, Y0s)
    Yerr = Y0s - mu_0t
    sigma = jnp.sqrt(jnp.einsum("n,nij->ij", weights, Yerr**2)).mean() * sigma
    sigma = jnp.maximum(sigma, 1e-3)
    return mu_0tm1, sigma


@jax.jit
def cem_update(weights, Y0s, sigma, mu_0t):
    idx = jnp.argsort(weights)[::-1][:10]
    mu_0tm1 = jnp.mean(Y0s[idx], axis=0)
    return mu_0tm1, sigma


def run_path_integral(args: Args):

    rng = jax.random.PRNGKey(seed=args.seed)

    update_fn = {
        "mppi": softmax_update,
        "cma-es": cma_es_update,
        "cem": cem_update,
    }[args.method]

    ## setup env

    env = d4orm.envs.get_env(args.env_name, args.Nagent)
    Nx = env.observation_size
    Nu = env.action_size
    # env functions
    step_env_jit = jax.jit(env.step)
    reset_env_jit = jax.jit(env.reset)
    rollout_env_jit = jax.jit(env.rollout)

    path = f"{d4orm.__path__[0]}/../results/{args.method}/{args.env_name}"
    os.makedirs(path, exist_ok=True)

    rng, rng_reset = jax.random.split(rng)  # NOTE: rng_reset should never be changed.
    state_init = reset_env_jit(rng_reset)

    ## run path interal

    mu_0T = jnp.zeros([args.Hsample, Nu])

    @jax.jit
    def update_once(carry, unused):
        t, rng, mu_0t, sigma = carry

        # sample from q_i
        rng, Y0s_rng = jax.random.split(rng)
        eps_u = jax.random.normal(Y0s_rng, (args.Nsample, args.Hsample, Nu)) * sigma
        Y0s = eps_u + mu_0t

        # esitimate mu_0tm1
        rewss, _, _, _ = jax.vmap(rollout_env_jit, in_axes=(None, None, 0))(state_init, env.xg, Y0s)

        # rewss = rewss.mean(axis=1)
        rews = rewss.mean(axis=-1)
        logp0 = (rews - rews.mean()) / rews.std() / args.temp_sample
        weights = jax.nn.softmax(logp0)
        mu_0tm1, sigma = update_fn(weights, Y0s, sigma, mu_0t)

        return (t - 1, rng, mu_0tm1, sigma), rews.mean()
    
    def update(mu_0T, rng, Niteration):
        sigma = 1.0
        mu_0t = mu_0T
        mu_0ts = []
        idx = 0
        rew_lst = [0.0]
        success_step = 0
        for idx in range(Niteration):
            steps_per_itr = 100
            for j in range(steps_per_itr):
                carry_once = (1, rng, mu_0t, sigma)
                (_, rng, mu_0t, sigma), rew = update_once(carry_once, None)

            mu_0ts.append(mu_0t)
            rews, _, goal_masks, collisions = rollout_env_jit(state_init, env.xg, mu_0t)
            rew_lst.append(round(float(rews.mean()), 4))

            if jnp.all(goal_masks[-1, :]) & jnp.all(collisions == 0) and success_step == 0:
                success_step = idx + 1
                break

        return mu_0ts, rew_lst, success_step

    rng, rng_warmup = jax.random.split(rng)

    # For warm-start
    mu_0ts, _, _ = update(mu_0T, rng_warmup, 1)

    for seed in range(1):
        rng = jax.random.PRNGKey(seed=seed+args.seed)
        rng, _ = jax.random.split(rng)
    
        start_time = time.time()  ###################################################################################

        mu_0ts, rew_lst, num_itr = update(mu_0T, rng, args.Niteration)

        end_time = time.time()  ###################################################################################
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.6f} seconds")

        mu_0ts = jnp.array(mu_0ts)
        xs = jnp.array([state_init.pipeline_state])
        collision, result = 0, 0
        state = state_init
        max_distances = jnp.linalg.norm(state.pipeline_state.reshape(args.Nagent, -1) - env.xg.reshape(args.Nagent, -1), axis=1)
        for t in range(mu_0ts[-1].shape[0]):
            state = step_env_jit(state, env.xg, mu_0ts[-1][t], max_distances)
            xs = jnp.concatenate([xs, state.pipeline_state[None]], axis=0)

            if not jnp.all(state.collision == 0):
                collision = 1
            result = int(jnp.count_nonzero(state.mask)) / int(state.mask.size)

        if collision: result = -1.0
        print("number of iterations:", num_itr)
        print("success rate:", result)

        rewss_final, _ , masks, _ = rollout_env_jit(state_init, env.xg, mu_0ts[-1])
        if result == 1.0:
            avg_steps_to_goal = jnp.sum(masks == 0) // args.Nagent
            max_steps_to_goal = jnp.max(jnp.argmax(masks, axis=0))
        else:
            avg_steps_to_goal = -1
            max_steps_to_goal = -1
        rew_final = rewss_final.mean()

        if args.save_images:
            filename = os.path.join(path, f"{args.method}_{args.env_name}_movement.gif")
            filename2 = os.path.join(path, f"{args.method}_{args.env_name}_trajectory.png")
            env.render_gif(xs, filename, filename2)
            if args.env_name in {"multi3dholo"}:
                env.render_gif_interactive(xs)

        def encode_float(val):
            if isinstance(val, float):
                s = f"{val:.0e}"  # always scientific notation, e.g., 4.0e-01
                s = s.replace('e-', 'em').replace('e+', 'ep')
                return s
            return str(val)

        def encode_args_for_filename(args: Args):
            keys = ['Nagent', 'Nsample', 'Hsample', 'temp_sample']
            return "_".join(encode_float(getattr(args, k)) for k in keys)

        if args.save_data:
            filename = os.path.join(path, f"runtime_data/{encode_args_for_filename(args)}.txt")
            os.makedirs(os.path.dirname(filename), exist_ok=True)  # Ensure the directory exists
            with open(filename, 'a') as f:
                f.write(f"{result}, {elapsed_time:.6f}, {num_itr}, {rew_final}, {avg_steps_to_goal}, {max_steps_to_goal}\n")

    return rew_final


if __name__ == "__main__":
    rew = run_path_integral(args=tyro.cli(Args))
    print(f"rew: {rew:.2e}")
