import numpy as np
import torch

import os

from utils.policy_robosuite_utils import make, ActorSAC, Actor, ActorCustomInit, ActorBCManiSkill
from utils.registration_utils import standardize_quaternion

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
policies_dir = os.path.join(parent_dir, "policies")

default_policy_type_dict = {
                            "Reach": "with_gripper",
                            "Lift": "irl_noquat",
                            "PickPlace": "irl",
                            "Stack": "irl",
                            }


save_idx = 1
def run_task(task, env, agent, num_episodes, render=False, save_obs=False, standardize_quat=False):
    global save_idx

    avg_errors_mm = []
    success_count = 0

    if task == "Lift":
        obj_pos_key = 'cube_pos'
    elif task == "PickPlace":
        obj_pos_key = 'Bread_pos'
    elif task == "Stack":
        obj_pos_key = 'cubeA_pos'
    elif task == "Reach":
        obj_pos_key = 'target_pos'

    obs_taskname = task.lower()
    if "bc" in policy_params:
        obs_taskname += "_bc"
    
    for i in range(num_episodes):
        obs = env.reset()

        obs_list = [obs]
        start_time = env.unwrapped.sim.data.time
        time_list = [start_time]
        actions = [np.zeros(7)]
        eef_pos_list = [env.observation_spec()['robot0_eef_pos']]
        obj_pos_list = [env.observation_spec()[obj_pos_key]]
        
        # for k,v in env.observation_spec().items():
        #     print(f"{k}: {v.shape}")
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            if standardize_quat and task=="Lift":
                obs[22:26] = standardize_quaternion(obs[22:26])

            action = agent.sample_action(obs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward   
            steps += 1
            
            if render:
                env.render()
            
            if save_obs:
                obs_list.append(obs)
                time_list.append(env.unwrapped.sim.data.time - start_time)
                actions.append(action)
                eef_pos_list.append(env.observation_spec()['robot0_eef_pos'])
                obj_pos_list.append(env.observation_spec()[obj_pos_key])

        if save_obs:
            if not os.path.exists("outputs/compare_sim_real"):
                os.makedirs("outputs/compare_sim_real")
            np.savez(f"outputs/compare_sim_real/obs_rbs_{obs_taskname}_{save_idx}.npz", obs_list=obs_list, time_list=time_list, actions=actions, obj_pos_list=obj_pos_list, eef_pos_list=eef_pos_list)
            save_idx += 1
        
        print(f"\nEpisode {i+1} | Success : {env._check_success()} | Length: {steps} | Reward: {episode_reward}")
        if task == "Reach":
            print(f"Distance to target: {obs[-3:]*1000} mm")
            avg_errors_mm.append(obs[-3:]*1000)
        if task == "Lift":
            print(f"Final distance to target: {np.linalg.norm(env.observation_spec()['cube_to_target_pos'])*1000:.3f} mm")
        if env._check_success():
            success_count += 1
        
    if task == "Reach":
        print(f"Average errors across {num_episodes} episodes:")
        print(f"Error: {np.mean(avg_errors_mm, axis=0)} mm, Error Norm: {np.linalg.norm(np.mean(avg_errors_mm, axis=0))} mm")
    print(f"\nSuccess rate: {success_count}/{num_episodes} = {100*success_count/num_episodes}%\n")
    
    if save_obs:
        print(f"Saved {len(obs_list)} observations to outputs/compare_sim_real/obs_rbs_{obs_taskname}_<index>.npz")


def test_imitation_robosuite(task, policy_type, render=True, save_obs=False, num_episodes=5):
    """
        task: Reach/Lift/PickPlace/Stack
        policy_type: None for default, else of format: (no_gripper/with_gripper) or (rl/irl)_(noquat/posquat/allquat)[_nogripwidth][_other_parmas]
    """
    global policy_params

    if policy_type == "":
        policy_type = default_policy_type_dict[task]
    policy_params = policy_type.split("_")

    if task != "Reach":
        assert policy_params[0] in ["rl", "irl", "bc"], "policy_type must start with rl or irl or bc"

    env_name = task
    robot = "xArm6"
    controller_type = "JOINT_VELOCITY"
    env_kwargs = {
        "horizon": 1000, 
        "gripper_types": "Robotiq85TouchGripper",
        "use_touch_obs": True,
        "base_types": "NullMount"
    }

    robot_keys = ['robot0_joint_pos_cos', 'robot0_joint_pos_sin']

    if task == "Reach":
        if "bc" in policy_params:
            object_keys = ['target_pos']
            env_kwargs.update({"table_full_size": (0.6, 0.6, 0.05)})
        else:
            robot_keys += ['robot0_joint_vel']
            object_keys = ['target_to_robot0_eef_pos']
            assert '_'.join(policy_params[:2]) in ["no_gripper", "with_gripper"]
            if "no_gripper" in policy_type:
                env_kwargs["gripper_types"] = None
            del env_kwargs["use_touch_obs"]
    elif task == "Lift":
        object_keys = ['robot0_gripper_width',
                        'cube_to_robot0_eef_pos',
                        'cube_to_robot0_eef_quat',
                        'cube_to_target_pos',
                        'robot0_touch']
        env_kwargs.update({"table_offset": (0., 0., 0.908)})
        if "bc" in policy_params:
            object_keys.remove('cube_to_robot0_eef_quat')
            # # FOR DEBUGGING
            # env_kwargs.update(init_cube_pos=np.array([0.0012, -0.0047,  0.9415]))
            if "mnskill" in policy_params:
                robot_keys += ['robot0_joint_vel']
                object_keys.remove('robot0_gripper_width')
                object_keys.remove('robot0_touch')
                object_keys.insert(0, 'is_grasped')
                env_kwargs.update({"base_types": "NullMount"})
        else:
            robot_keys += ['robot0_joint_vel']
            assert policy_params[1] in ["noquat", "posquat", "allquat"], "policy_type must contain noquat, posquat, or allquat"
            if "noquat" in policy_params:
                object_keys.remove('cube_to_robot0_eef_quat')
            if "nogripwidth" in policy_params:
                object_keys.remove('robot0_gripper_width')
    elif task == "PickPlace":
        env_name = "PickPlaceBread"
        object_keys = [  
            'robot0_gripper_width',
            'Bread_to_robot0_eef_pos',
            'Bread_to_Bread_bin_pos',
            'robot0_touch',
        ]
        env_kwargs.update(bin1_pos=(-0.05, -0.25, 0.90), 
                            bin2_pos=(-0.05, 0.28, 0.90))
        if "bc" not in policy_params:
            robot_keys += ['robot0_joint_vel']
            object_keys.remove('robot0_gripper_width')
    elif task == "Stack":
        object_keys = [  
            'robot0_gripper_width',
            'gripper_to_cubeA',
            'gripper_to_cubeB',
            'cubeA_to_cubeB',
            'robot0_touch',
        ]
        # del env_kwargs["use_touch_obs"]
        if "bc" in policy_params:
            object_keys.remove('gripper_to_cubeB')
            env_kwargs.update({"table_offset": (0., 0., 0.908)})
        else:
            robot_keys += ['robot0_joint_vel']
            object_keys.remove('robot0_gripper_width')
    obs_keys = robot_keys + object_keys

    if "bc" in policy_params:
        env_kwargs.update(initialization_noise=None)

    env = make(env_name=env_name, robots=robot, controller_type=controller_type, render=render,
                obs_keys=obs_keys, **env_kwargs)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    print(f"\n\nobs_dim: {obs_dim}, act_dim: {act_dim}\n\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if task=="Reach":
        if policy_type == "no_gripper":
            policy = ActorSAC(obs_dim, act_dim, 3, 512, device).to(device)
        elif policy_type == "with_gripper":
            policy = Actor(obs_dim, act_dim, 3, 512, device).to(device)
        else:
            print(f"Invalid policy type {policy_type} for Reach task")
            return
        policy.load_state_dict(torch.load(policies_dir + f"/reach_actor_{policy_type}.pt"))
    else:
        if "irl" in policy_params:
            policy = ActorSAC(obs_dim, act_dim, 3, 512, device).to(device)
            policy.load_state_dict(torch.load(policies_dir + f"/{task.lower()}_actor_{policy_type}.pt"))
        elif "rl" in policy_params:
            policy = Actor(obs_dim, act_dim, 3, 512, device).to(device)
            policy.load_state_dict(torch.load(policies_dir + f"/{task.lower()}_actor_{policy_type}.pt"))
        elif "bc" in policy_params:
            policy = ActorCustomInit(obs_dim, act_dim, 3, 256).to(device)
            if "mnskill" in policy_params:
                policy = ActorBCManiSkill(obs_dim, act_dim).to(device)
            policy.load_state_dict(torch.load(policies_dir + f"/{task.lower()}_actor_{policy_type}.pt")["actor"])

    print(f"Testing policy type: {policy_type}")
    run_task(task, env, policy, num_episodes, render=render, save_obs=save_obs, standardize_quat=("posquat" in policy_params))

    env.close()


if __name__ == '__main__':
    task = "PickPlace"
    policy_type = None

    print(f"\nTesting {task} task\n\n")    
    test_imitation_robosuite(task, policy_type, render=False, save_obs=False, num_episodes=20)
