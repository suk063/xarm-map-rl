import pathlib
import datetime
import time
import uuid
import io

import numpy as np
from tqdm.auto import tqdm

import robosuite as suite
from policies.human_policy import ReachPolicy, LiftPolicy, BaseHumanPolicy
from envs.reach import Reach
from utils.debug_utils import format_ndarray

def make_robosuite_env(
    env_name, 
    robots="Panda", 
    controller_type='OSC_POSE', 
    render=False,
    offscreen_render=False,
    **kwargs
):
    controller_configs = suite.load_controller_config(default_controller=controller_type)
    env = suite.make(
        env_name=env_name, # try with other tasks like "Stack" and "Door"
        robots=robots,  # try with other robots like "Sawyer" and "Jaco"
        reward_shaping=True,
        has_renderer=render,
        has_offscreen_renderer=offscreen_render,
        use_camera_obs=offscreen_render,
        use_object_obs=True,
        controller_configs=controller_configs,
        **kwargs,
    )
    return env


def eplen(episode):
    return len(episode['action'])

def sample_episodes(env, policy, directory, num_episodes=1, policy_obs_keys=None, render=False):
    # Save all observation keys from environment
    episodes_saved = 0
    while episodes_saved < num_episodes:
        obs = env.reset()

        if isinstance(policy, BaseHumanPolicy):
            policy.reset()

        obs_keys = list(obs.keys())
        done = False
        episode = {}
        for k in obs_keys:
            episode[k] = [obs[k]]
        episode['action'] = []
        episode['reward'] = []

        while not done:       
            if policy_obs_keys is not None:  
                policy_obs = np.concatenate([obs[k] for k in policy_obs_keys])
            else:
                policy_obs = obs
            action, _ = policy.predict(policy_obs)
            obs, rew, done, info = env.step(action)

            if render:
                env.render()

            for k in obs_keys:
                episode[k].append(obs[k])
            episode['action'].append(action)
            episode['reward'].append(rew)

        print(f"Episode return: {np.sum(episode['reward']):.2f}")
        save_episode(directory, episode)
        episodes_saved += 1
    env.close()


def save_episode(directory, episode):
    timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    identifier = str(uuid.uuid4().hex)
    length = eplen(episode)
    filename = directory / f'{timestamp}-{identifier}-{length}.npz'
    with io.BytesIO() as f1:
        np.savez_compressed(f1, **episode)
        f1.seek(0)
        with filename.open('wb') as f2:
            f2.write(f1.read())
    return filename


def save_osc_episodes(num_episodes=64, render=False):

    env_name = "Reach"
    # env_name = "Lift"
    # controller_type = "OSC_POSE"
    controller_type = "OSC_POSITION"
    robots = "Panda"
    # robots = "Sawyer"
    # robots = "xArm6"

    noise = True

    if env_name == "Reach":
        policy_cls = ReachPolicy
        env_kwargs = {"horizon": 200}
    if env_name == "Lift":
        policy_cls = LiftPolicy
        env_kwargs = {"horizon": 200, "use_touch_obs": True}
        if robots == "Panda":
            env_kwargs["gripper_types"] = 'PandaTouchGripper'
        elif robots == "xArm6":
            env_kwargs["gripper_types"] = 'Robotiq85TouchGripper'

    env = make_robosuite_env(env_name, robots, controller_type, render=render, **env_kwargs)
    policy = policy_cls(env)

    directory = pathlib.Path(f"./human_demonstrations_noise_new/{env_name}/{robots}/{controller_type}")
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=False)

    episodes_saved = 0
    while episodes_saved < num_episodes:
        obs = env.reset()

        if isinstance(policy, BaseHumanPolicy):
            policy.reset()

        obs_keys = list(obs.keys())
        done = False
        episode = {}
        for k in obs_keys:
            episode[k] = [obs[k]]
        episode['action'] = []
        episode['reward'] = []

        print(obs['robot0_eef_pos'])

        while not done:       
            # action, _ = policy.predict(obs)
            # if noise:
            #     action += np.random.randn(*action.shape) * 0.5
            #     action = np.clip(action, -1, 1)
                # action = np.zeros_like(action)
            action = np.random.uniform(low=-1, high=1, size=env.action_dim)
            action[0] += 0.1
            action[3:] = 0
            obs, rew, done, info = env.step(action)

            if render:
                env.render()

            for k in obs_keys:
                episode[k].append(obs[k])
            episode['action'].append(action)
            episode['reward'].append(rew)

        print(episode['robot0_eef_pos'][-1])
        print(f"Episode return: {np.sum(episode['reward']):.2f}")
        save_episode(directory, episode)
        episodes_saved += 1
    env.close()

def save_jv_episodes(num_episodes=64, render=False):

    env_name = "Reach"
    # env_name = "Lift"
    controller_type = "JOINT_VELOCITY"
    robots = "Panda"
    # robots = "Sawyer"
    # robots = "xArm6"

    noise = True

    if env_name == "Reach":
        policy_cls = ReachPolicy
        env_kwargs = {"horizon": 200}
    if env_name == "Lift":
        policy_cls = LiftPolicy
        env_kwargs = {"horizon": 200, "use_touch_obs": True}
        if robots == "Panda":
            env_kwargs["gripper_types"] = 'PandaTouchGripper'
        elif robots == "xArm6":
            env_kwargs["gripper_types"] = 'Robotiq85TouchGripper'

    env = make_robosuite_env(env_name, robots, controller_type, render=render, **env_kwargs)
    policy = policy_cls(env)

    directory = pathlib.Path(f"./human_demonstrations_noise_new/{env_name}/{robots}/{controller_type}")
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=False)

    episodes_saved = 0
    while episodes_saved < num_episodes:
        obs = env.reset()

        if isinstance(policy, BaseHumanPolicy):
            policy.reset()

        obs_keys = list(obs.keys())
        done = False
        episode = {}
        for k in obs_keys:
            episode[k] = [obs[k]]
        episode['action'] = []
        episode['reward'] = []

        print(obs['robot0_eef_pos'])

        while not done:       
            # action, _ = policy.predict(obs)
            action = np.random.uniform(low=-1, high=1, size=env.action_dim)
            action[0] += 0.1
            action[3:] = 0
            obs, rew, done, info = env.step(action)

            if render:
                env.render()

            for k in obs_keys:
                episode[k].append(obs[k])
            episode['action'].append(action)
            episode['reward'].append(rew)

        print(episode['robot0_eef_pos'][-1])
        print(f"Episode return: {np.sum(episode['reward']):.2f}")
        save_episode(directory, episode)
        episodes_saved += 1
    env.close()

def osc_to_jv(num_episodes=64, render=False):
    """
    Use osc to find waypoint joint position
    """

    env_name = "Reach"
    # env_name = "Lift"
    # robots = "Panda"
    robots = "Sawyer"
    # robots = "xArm6"

    if env_name == "Reach":
        policy_cls = ReachPolicy
        env_kwargs = {"horizon": 200}
    if env_name == "Lift":
        policy_cls = LiftPolicy
        env_kwargs = {"horizon": 200, "use_touch_obs": True,}
        if robots == "Panda":
            env_kwargs["gripper_types"] = 'PandaTouchGripper'
        elif robots == "xArm6":
            env_kwargs["gripper_types"] = 'Robotiq85TouchGripper'

    # Intialize an env with OSC_POSE controller
    env = make_robosuite_env(env_name, robots, "OSC_POSE", render=render, **env_kwargs)
    policy = policy_cls(env)

    # Initialize an env with JOINT_VELOCITY controller to track the osc_pose episode
    jv_env = make_robosuite_env(
        env_name, 
        robots=robots, 
        controller_type="JOINT_VELOCITY", 
        render=render, 
        **env_kwargs
    )

    directory = pathlib.Path(f"./human_demonstrations_noise_new/{env_name}/{robots}/JOINT_VELOCITY")
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=False)

    episodes_saved = 0
    while episodes_saved < num_episodes:

        obs = env.reset()
        policy.reset()
        target_pos = np.copy(env.target_pos)    

        # Record the mujoco states so that we can replay later
        task_xml = env.sim.model.get_xml()
        task_init_state = np.array(env.sim.get_state().flatten())   

        desired_jps, gripper_actions, action_infos = [], [], []
        done = False
        # while not done:
        for i in range(200):
            # action, action_info = policy.predict(obs)
            action = np.random.uniform(low=-1, high=1, size=env.action_dim)
            action[0] += 0.08
            action[3:] = 0
            obs, rew, done, info = env.step(action)

            desired_jps.append(env.robots[0]._joint_positions)
            gripper_actions.append(action[-1])
            # action_infos.append(action_info)
            if render:
                env.render()

        # Reset environment to the same initial state
        jv_env.reset()
        jv_env.reset_from_xml_string(task_xml)
        jv_env.sim.reset()
        jv_env.sim.set_state_from_flattened(task_init_state)
        jv_env.sim.forward()   

        obs = jv_env._get_observations(force_update=True)
        obs_keys = list(obs.keys())
        episode = {}
        for k in obs_keys:
            episode[k] = [obs[k]]
        episode['action'] = []
        episode['reward'] = []    

        # for i, (next_jp, gripper_action, action_info) in enumerate(
        #     zip(desired_jps, gripper_actions, action_infos)):
        for i, (next_jp, gripper_action) in enumerate(
            zip(desired_jps, gripper_actions)):

            action = np.zeros(jv_env.robots[0].dof)
            action[-1] = gripper_action
            err = next_jp - jv_env.robots[0]._joint_positions

            # if action_info['stage'] in {'lower_gripper', 'lift_gripper'}:
            #     kp = 20
            # else:
            #     kp = 2
            kp = 2
            action[:-1] = np.clip(err*kp, -1, 1)

            obs, rew, done, info = jv_env.step(action)
            if render:
                jv_env.render()

            for k in obs_keys:
                episode[k].append(obs[k])
            episode['action'].append(action)
            episode['reward'].append(rew)

        print(f"Episode {episodes_saved} return: {np.sum(episode['reward']):.2f}")
        print(episode['robot0_eef_pos'][-1])
        save_episode(directory, episode)
        episodes_saved += 1        
        # if np.sum(episode['reward']) > 50:
        #     save_episode(directory, episode)
        #     episodes_saved += 1

    env.close()
    jv_env.close()


def test_xArm6():

    robots = "xArm6"
    env_name = "Reach"
    render = True

    if env_name == "Reach":
        policy_cls = ReachPolicy
        env_kwargs = {"horizon": 200}
    if env_name == "Lift":
        policy_cls = LiftPolicy
        env_kwargs = {"horizon": 200, "use_touch_obs": True,}
        if robots == "Panda":
            env_kwargs["gripper_types"] = 'PandaTouchGripper'
        elif robots == "xArm6":
            env_kwargs["gripper_types"] = 'Robotiq85TouchGripper'

    # Intialize an env with OSC_POSE controller
    env = make_robosuite_env(env_name, robots, "OSC_POSE", render=render, **env_kwargs)
    policy = policy_cls(env)


def check_eef_within_limits(pos):
    '''
        Check if the end effector is within the limits of the workspace.
        Return 1 if outside the outher limit, -1 if inside the inner limit, 0 if okay.
    '''
    x, y, z = pos

    # Inner limits
    if x**2 + y**2 < 0.2**2 and z < 0.1:
        return -1
    # Outer limits
    if x**2 + y**2 + z**2 > 0.65**2:
        return 1
    return 0


def save_xArm6_eefpos_to_jv_episodes(num_episodes=2, horizon=5):
    env_name = "Reach"
    controller_type = "JOINT_VELOCITY"
    robots = "xArm6"

    env = Reach(mode=0, simulated=True, target_src=np.zeros(3), has_gripper=True)
    jv_env = Reach(mode=4, simulated=True, target_src=np.zeros(3), has_gripper=True)
    episodes_saved = 0
    obs_keys = ["robot0_joint_pos_cos", "robot0_joint_pos_sin", "robot0_joint_vel", "robot0_eef_pos"]
    robosuite_robot_base_pos = np.array([-0.35,  0, 0.7])
    directory = pathlib.Path(f"./human_demonstrations_noise_new/{env_name}/{robots}/{controller_type}")
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=False)

    while episodes_saved < num_episodes:
        obs = env.reset()
        start_time = time.time()
        prev_time = start_time

        i = 0
        desired_jps = []
        durations = []

        # First run policy using EEF position control. Save joint angles along the way
        while i < horizon:
            print(f"Episode {episodes_saved+1} (EEF pos control) | Step {i+1} | EEF pos: {obs['robot0_eef_pos']}")
            eef_pos_delta_action = np.random.uniform(low=-1, high=1, size=6)
            if obs['robot0_eef_pos'][0] < 0.2:
                eef_pos_delta_action[0] = max(0, eef_pos_delta_action[0])
            if obs['robot0_eef_pos'][2] < -0.2:
                eef_pos_delta_action[2] = max(0, eef_pos_delta_action[2])
            eef_pos_delta_action[0] += 0.1
            eef_pos_delta_action[3:] = 0
            
            actual_eef_pos_delta = eef_pos_delta_action * 0.05
            
            action = np.append(100 * actual_eef_pos_delta, 0)
            obs, rew, done, info = env.step(action)

            desired_jps.append(obs['robot0_joint_pos'])
            durations.append(time.time() - prev_time)
            prev_time = time.time()
            i += 1

        # Now run same trajectory using joint angle differences for joint velocity control. Save episodes.
        obs = jv_env.reset()
        
        episode = {}
        for k in obs_keys:
            episode[k] = [obs[k]]
        episode["robot0_eef_pos"][-1] += robosuite_robot_base_pos

        episode['action'] = []
        episode['reward'] = []

        i = 0
        for desired_jp, duration in zip(desired_jps, durations):
            print(f"Episode {episodes_saved+1} (Converted JV Control) | Step {i+1} | EEF pos: {obs['robot0_eef_pos'] - robosuite_robot_base_pos}")

            action = np.zeros(7)
            action[-1] = -1
            err = desired_jp - obs['robot0_joint_pos']

            kp = 0.5
            action[:-1] = np.clip(err*kp, -1, 1)

            obs, rew, done, info = jv_env.step(action, duration=duration)

            for k in obs_keys:
                episode[k].append(obs[k])
            episode["robot0_eef_pos"][-1] += robosuite_robot_base_pos
            episode['action'].append(action)
            episode['reward'].append(rew)
            i += 1

        print(f"Episode return: {np.sum(episode['reward']):.2f} | Saving took {time.time() - start_time} seconds.\n")
        save_episode(directory, episode)
        episodes_saved += 1


def save_xArm6_direct_jv_episodes(num_episodes=2, horizon=5):
    env_name = "Reach"
    controller_type = "JOINT_VELOCITY"
    robots = "xArm6"

    env = Reach(mode=4, simulated=True, target_src=np.array([0.56, 0, 0.088]), has_gripper=True)
    env.xarm.arm.set_self_collision_detection(False)
    env.xarm.arm.set_collision_rebound(True)

    episodes_saved = 0
    obs_keys = ["robot0_joint_pos_cos", "robot0_joint_pos_sin", "robot0_joint_vel", "robot0_eef_pos"]
    robosuite_robot_base_pos = np.array([-0.35,  0, 0.7])
    directory = pathlib.Path(f"./human_demonstrations_noise_new/{env_name}/{robots}/{controller_type}")
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=False)

    running_mean = np.zeros(3)
    while episodes_saved < num_episodes:
        obs = env.reset()

        episode = {}
        for k in obs_keys:
            episode[k] = [obs[k]]
        episode["robot0_eef_pos"][-1] += robosuite_robot_base_pos

        episode['action'] = []
        episode['reward'] = []

        for i in range(horizon):
            jv_actions = np.random.uniform(low=-1, high=1, size=6)
            # jv_actions[0] /= 2
            if np.random.rand() < 0.66:
                # jv_actions[1] += 0.22
                # jv_actions[2] -= 0.32
                jv_actions[1] = max(0, jv_actions[1])
                jv_actions[2] = min(0, jv_actions[2]) - 0.28
            if check_eef_within_limits(obs['robot0_eef_pos'] - robosuite_robot_base_pos) < 0:
                jv_actions[1] = max(0, jv_actions[1])
                jv_actions[2] = min(0, jv_actions[2])
            jv_actions[5] = 0
            
            action = np.append(0.5 * jv_actions, -1)
            obs, rew, done, info = env.step(action)
            
            for k in obs_keys:
                episode[k].append(obs[k])
            episode["robot0_eef_pos"][-1] += robosuite_robot_base_pos
            episode['action'].append(action)
            episode['reward'].append(rew)

        print(f"Episode {episodes_saved+1} | Return: {np.sum(episode['reward']):.2f} | Mean EEF pos: {format_ndarray(np.mean(episode['robot0_eef_pos'], axis=0))} | STD EEF pos: {format_ndarray(np.std(episode['robot0_eef_pos'], axis=0))} | Final EEF pos: {format_ndarray(obs['robot0_eef_pos'] - robosuite_robot_base_pos)}")
        save_episode(directory, episode)
        episodes_saved += 1

        running_mean += np.mean(episode['robot0_eef_pos'], axis=0)
        print(f"Running mean: {format_ndarray(running_mean / episodes_saved)}\n")



if __name__ == '__main__':
    # save_osc_episodes(num_episodes=1000, render=True)
    # save_jv_episodes(num_episodes=100, render=False)
    # osc_to_jv(num_episodes=2000, render=False)
    
    # test_xArm6()
    # save_xArm6_eefpos_to_jv_episodes(num_episodes=1000, horizon=200)
    save_xArm6_direct_jv_episodes(num_episodes=1000, horizon=200)