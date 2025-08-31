import numpy as np
import torch

from utils.policy_robosuite_utils import make_robosuite_env, build_mlp, ActorCustomInit
from utils.debug_utils import format_ndarray
from envs.reach import Reach
from envs.lift import Lift
from envs.pick_place import PickPlace

import os
from functools import partial
import time
np.set_printoptions(precision=4, suppress=True)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
policies_dir = os.path.join(parent_dir, "policies")

xarm_base_pos_rbs = np.array([-0.407, 0., 1.083])       # xarm robot base position in robosuite

save_idx = 1
def evaluate_robosuite(task, env, obs_enc, act_dec, lat_actor, render=False, save_obs=False):
    global save_idx
    
    obs = env.reset()
    done = False
    episode_reward = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if task == "TransferLift":
        obj_pos_key = 'cube_pos'
    elif task == "TransferPickPlace":
        obj_pos_key = 'Bread_pos'
    # elif task == "TransferStack":
    #     obj_pos_key = 'cubeA_pos'
    elif task == "TransferReach" or task == "TransferTrack":
        obj_pos_key = 'target_pos'

    obs_list = []
    time_list = []
    actions = []
    eef_pos_list = []
    obj_pos_list = []
    start_time = env.sim.data.time

    # print(f"robot base to cube: {obs['cube_pos'] - env.robots[0].base_pos}")
    # while not done:
    for i in range(200):
        with torch.no_grad():
            robot_obs = np.concatenate([obs[k] for k in robot_keys])
            robot_obs = torch.from_numpy(robot_obs).float().unsqueeze(0).to(device)
            lat_obs = obs_enc(robot_obs)
            obj_obs = np.concatenate([obs[k] for k in object_keys])
            obj_obs = torch.from_numpy(obj_obs).float().unsqueeze(0).to(device)
            policy_obs = torch.cat([lat_obs, obj_obs], dim=-1)
            lat_act = lat_actor(policy_obs)
            robot_act, gripper_act = lat_act[:, :-1], lat_act[:, -1].unsqueeze(dim=-1)
            action = act_dec(torch.cat([robot_act, robot_obs], dim=-1))
            action = torch.cat([action, gripper_act], dim=-1)

        action = action.cpu().data.numpy().flatten()
        # print(action)

        obs, reward, done, _ = env.step(action)
        episode_reward += reward

        if render:
            env.render()
        
        if save_obs:
            obs_list.append(np.concatenate([obs[k] for k in keys]))
            time_list.append(env.sim.data.time - start_time)
            actions.append(action)
            eef_pos_list.append(env.observation_spec()['robot0_eef_pos'])
            obj_pos_list.append(env.observation_spec()[obj_pos_key])

    if save_obs:
        if not os.path.exists("outputs/compare_sim_real"):
            os.makedirs("outputs/compare_sim_real")
        np.savez(f"outputs/compare_sim_real/obs_rbs_{task.lower()}_{save_idx}.npz", obs_list=obs_list, time_list=time_list, actions=actions, obj_pos_list=obj_pos_list, eef_pos_list=eef_pos_list)
        save_idx += 1
    

    print(f"task: {task}, success: {env._check_success()}, episode reward: {episode_reward:.3f}")
    if task == "TransferReach":
        dist = np.linalg.norm(obs['target_to_robot0_eef_pos'])
        print(f"Distance to target: {dist*1000:.3f} mm")
        return episode_reward, dist
    else:
        return episode_reward, None


# TODO: Implement save_obs
def evaluate_xarm_reach(lat_actor, obs_enc, act_dec, n_episodes=5, simulate_xarm=True, debug=False, save_obs=False, episodic=True, target_src="random"):
    if target_src == "random":
        # We have target pos and init eef in robosuite frame, and init eef in xarm frame
        # So target_xarm = target_robosuite - init_eef_robosuite + init_eef_xarm
        target_pos_center = np.array([0.05, 0, 1]) - np.array([-0.2, 0, 1.05]) + np.array([0.207, 0, -0.033])
        # target_pos = np.random.uniform(low=target_pos_center-0.1, high=target_pos_center+0.1)
        target_pos = np.random.uniform(low=target_pos_center - np.array([0.1, 0.25, 0]), high=target_pos_center + np.array([0.1, 0.25, 0.1]))
        target_src = target_pos

    gripper_action = 1
    error_threshold = 0.02

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = Reach(mode=4, simulated=simulate_xarm, target_src=target_src, has_gripper=True)    

    mean_dist = 0
    for _ in range(n_episodes):
        obs = env.reset()
        dist = np.linalg.norm(obs['target_to_robot0_eef_pos'])
        done = False
        if target_src == "random":
            env.target_pos = np.random.uniform(low=target_pos_center-0.1, high=target_pos_center+0.1)
        target_pos_robosuite = env.target_pos + np.array([-0.2, 0, 1.05]) - np.array([0.207, 0, -0.033])

        for i in range(200):
            if not done:
                if debug:
                    print(f"Distance to target: {dist*1000} mm, target_pos: {obs['target_pos']*1000} mm")

                robot_obs = np.concatenate([obs[k] for k in robot_keys])
                robot_obs = torch.from_numpy(robot_obs).float().unsqueeze(0).to(device)
                lat_obs = obs_enc(robot_obs)
                obj_obs = np.concatenate([obs[k] for k in object_keys])
                obj_obs = torch.from_numpy(obj_obs).float().unsqueeze(0).to(device)
                policy_obs = torch.cat([lat_obs, obj_obs], dim=-1)
                lat_act = lat_actor(policy_obs)
                robot_act, gripper_act = lat_act[:, :-1], lat_act[:, -1].unsqueeze(dim=-1)
                action = act_dec(torch.cat([robot_act, robot_obs], dim=-1))
                action = torch.cat([action, gripper_act], dim=-1)
                action = action.cpu().data.numpy().flatten()
                
                action_scaled = action * 0.5
                obs, rew, done, info = env.step(np.append(action_scaled[:6], gripper_action))

            dist = np.linalg.norm(obs['target_to_robot0_eef_pos'])
            done = dist < error_threshold or i == 199
            if done:
                print(f"Distance to target: {dist*1000:.3f} mm, target_pos: {obs['target_pos']*1000} mm\n")
                if dist < error_threshold:
                    print("Target reached!")
                if episodic:
                    if i == 199:
                        print("Episode ends!")
                    mean_dist += dist
                    break
            else:
                gripper_action = 1
    env.reset()
    env.close()
    print(f"Mean distance to target: {mean_dist*1000/n_episodes} mm\n")



def evaluate_xarm_lift(lat_actor, obs_enc, act_dec, n_episodes=5, simulate_xarm=True, debug=False, save_obs=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = Lift(mode=4, simulated=simulate_xarm, use_pose=False)    

    obs_list = []
    time_list = []
    actions = []
    cube_pos_list = []
    eef_pos_list = []
    start_time = time.time()

    success_count = 0
    for _ in range(n_episodes):
        obs = env.reset()
        print(f"\nTASK: Attempting to lift cube at {obs['cube_pos']}, in robosuite cube_pos is {obs['cube_pos'] + np.array([-0.407, 0, 1.083])}")

        init_cube_height = obs['cube_pos'][2]
        use_gripper_policy = False
        enable_fake_touch = False


        for _ in range(200):        
            # print(obs['cube_pos'], obs['cube_to_target_pos'])
            # print(f"cube_to_target_pos: {obs['cube_to_target_pos']}, cube_pos: {obs['cube_pos']}") 
            # print(f"touch_obs: {obs['robot0_touch']}") 
            robot_obs = np.concatenate([obs[k] for k in robot_keys])
            robot_obs = torch.from_numpy(robot_obs).float().unsqueeze(0).to(device)
            lat_obs = obs_enc(robot_obs)
            obj_obs = np.concatenate([obs[k] for k in object_keys])
            obj_obs = torch.from_numpy(obj_obs).float().unsqueeze(0).to(device)
            policy_obs = torch.cat([lat_obs, obj_obs], dim=-1)
            lat_act = lat_actor(policy_obs)
            robot_act, gripper_act = lat_act[:, :-1], lat_act[:, -1].unsqueeze(dim=-1)
            action = act_dec(torch.cat([robot_act, robot_obs], dim=-1))
            action = torch.cat([action, gripper_act], dim=-1)
            action = action.cpu().data.numpy().flatten()
            # action[2] = min(0, action[2])

            action_scaled = action*0.5

            if save_obs:
                curr_time = time.time() - start_time
                time_list.append(curr_time)
                obs_list.append(np.concatenate([obs[k] for k in keys]))
                actions.append(action)
                cube_pos_list.append(obs['cube_pos'])
                eef_pos_list.append(obs['robot0_eef_pos'])

                if not os.path.exists("outputs/compare_sim_real"):
                    os.makedirs("outputs/compare_sim_real/")
                np.savez("outputs/compare_sim_real/obs_xarm_transferlift.npz", obs_list=obs_list, time_list=time_list, actions=actions, cube_pos_list=cube_pos_list, eef_pos_list=eef_pos_list)
            

            dist_cube_to_eef = np.linalg.norm(obs['cube_to_robot0_eef_pos'])
            dist_cube_to_target = np.linalg.norm(obs['cube_to_target_pos'])
            if dist_cube_to_eef < 0.08 or use_gripper_policy:
                use_gripper_policy = True
                obs, rew, done, info = env.step(action_scaled)
            else:
                obs, rew, done, info = env.step(np.append(action_scaled[:6], -1))
            done = done or (obs['cube_pos'][2] - init_cube_height > 0.3)

            obs["robot0_gripper_width"] = np.array([0.0904])
            obs["cube_to_robot0_eef_pos"] = obs["robot0_eef_pos"] - obs["cube_pos"]

            # Synthetic observations for simulated runs
            if simulate_xarm and (dist_cube_to_eef < 0.03 or enable_fake_touch):
                obs['robot0_touch'] = np.array([1, 1])
                obs['cube_to_robot0_eef_pos'] = np.zeros(3)
                obs['cube_to_target_pos'] = obs['target_pos'] - obs['robot0_eef_pos']
                if not enable_fake_touch:
                    print(f"Virtual cube grasped")
                    enable_fake_touch = True
                dist_cube_to_target = np.linalg.norm(obs['cube_to_target_pos'])
            if debug:
                print(f"Distance to cube: {obs['cube_to_robot0_eef_pos']*1000} mm, norm: {dist_cube_to_eef*1000} mm") 
            
            # Virtual gripper width to match robosuite observations
            if (obs["robot0_touch"] > 0.9).all():
                obs["robot0_gripper_width"] = np.array([0.0295])
            else:
                obs['robot0_gripper_width'] = np.array([0.0913])
            
            if done:
                print("\nCube Lifted!\n")
                success_count += 1
                break
            elif simulate_xarm and dist_cube_to_target < 0.03:
                print("\nVirtual Cube Lifted!\n")
                success_count += 1
                break
    env.reset()
    env.close()
    print(f"Success rate: {success_count/n_episodes*100:.2f}%\n")

def evaluate_xarm_pickplace(lat_actor, obs_enc, act_dec, n_episodes=5, simulate_xarm=True, debug=False, save_obs=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # target_pos = np.array([0.407, 0., -0.083])      # base frame
    env = PickPlace(mode=4, simulated=simulate_xarm, use_pose=False)    

    # Replace Bread with cube
    object_keys[1] = "cube_to_robot0_eef_pos"
    object_keys[2] = "cube_to_bin_pos"
    keys = robot_keys + object_keys

    obs_list = []
    time_list = []
    actions = []
    cube_pos_list = []
    eef_pos_list = []
    start_time = time.time()

    success_count = 0
    for _ in range(n_episodes):
        obs = env.reset()
        print(f"\nTASK: Attempting to lift cube at {obs['cube_pos']}, in robosuite cube_pos is {obs['cube_pos'] + np.array([-0.407, 0, 1.083])}")

        init_cube_height = obs['cube_pos'][2]
        use_gripper_policy = False
        enable_fake_touch = False
        print_info = True

        for i in range(300):        

            print(obs['cube_pos'], obs['cube_to_robot0_eef_pos'], obs['robot0_touch'], obs['robot0_gripper_width'], obs['cube_to_bin_pos'])
            print(env.cube_over_bin())
            # print(f"cube_to_target_pos: {obs['cube_to_target_pos']}, cube_pos: {obs['cube_pos']}") 
            # print(f"touch_obs: {obs['robot0_touch']}") 
            robot_obs = np.concatenate([obs[k] for k in robot_keys])
            robot_obs = torch.from_numpy(robot_obs).float().unsqueeze(0).to(device)
            lat_obs = obs_enc(robot_obs)
            obj_obs = np.concatenate([obs[k] for k in object_keys])
            obj_obs = torch.from_numpy(obj_obs).float().unsqueeze(0).to(device)
            policy_obs = torch.cat([lat_obs, obj_obs], dim=-1)
            lat_act = lat_actor(policy_obs)
            robot_act, gripper_act = lat_act[:, :-1], lat_act[:, -1].unsqueeze(dim=-1)
            action = act_dec(torch.cat([robot_act, robot_obs], dim=-1))
            action = torch.cat([action, gripper_act], dim=-1)
            action = action.cpu().data.numpy().flatten()

            action_scaled = action*0.3
            # print(action_scaled)

            if save_obs:
                curr_time = time.time() - start_time
                time_list.append(curr_time)
                obs_list.append(np.concatenate([obs[k] for k in keys]))
                actions.append(action)
                cube_pos_list.append(obs['cube_pos'])
                eef_pos_list.append(obs['robot0_eef_pos'])

                if not os.path.exists("outputs/compare_sim_real"):
                    os.makedirs("outputs/compare_sim_real/")
                np.savez("outputs/compare_sim_real/obs_xarm_transferpickplace.npz", obs_list=obs_list, time_list=time_list, actions=actions, cube_pos_list=cube_pos_list, eef_pos_list=eef_pos_list)
            

            dist_cube_to_eef = np.linalg.norm(obs['cube_to_robot0_eef_pos'])
            dist_cube_to_bin = np.linalg.norm(obs['cube_to_bin_pos'])
            dist_bin_to_eef = np.linalg.norm(obs['bin_to_robot0_eef_pos'])

            if use_gripper_policy:
                use_gripper_policy = True
                obs, rew, done, info = env.step(action_scaled)
            else:
                obs, rew, done, info = env.step(np.append(action_scaled[:6], -1))
            # done = done or (obs['cube_pos'][2] - init_cube_height > 0.3)

            # Logging task stages and creating synthetic observations for simulated runs
            if env.curr_stage == 1 and print_info:
                print(f"Reaching Cube | Distance of arm to cube: {format_ndarray(obs['cube_to_robot0_eef_pos']*1000)} mm, norm: {dist_cube_to_eef*1000} mm")
            if env.curr_stage == 2:
                # Stage 2: Bringing cube to bin and placing it in
                use_gripper_policy = True
                if print_info:
                    print(f"Bringing Cube to Bin | Distance of cube to bin: {format_ndarray(obs['cube_to_bin_pos']*1000)} mm, norm: {dist_cube_to_bin*1000}")
            if env.curr_stage == 3:
                # Stage 3: Lifting arm after placing cube
                use_gripper_policy = False
                if print_info:
                    print(f"Placing Cube in Bin | Touch: {obs['robot0_touch']}, Distance of arm to bin: {format_ndarray(obs['bin_to_robot0_eef_pos']*1000)} mm, norm: {dist_bin_to_eef*1000}")
            
            if done:
                print("\nCube placed in bin!\n")
                break


        env.reset()
    env.close()
    print(f"Success rate: {success_count/n_episodes*100:.2f}%\n")

def setup_transfer_envs(taskname):
    global robot_keys, object_keys, keys

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env_name = taskname
    render = True
    camera_name = "frontview"

    src_env_kwargs = {"horizon": 200, 
        "offscreen_render": True, "camera_names": camera_name,
        "camera_heights": 512, "camera_widths": 512,
        "gripper_types": "PandaTouchGripper",
        "ignore_done": True}
        # "init_cube_pos": [0, 0]}
    tgt_env_kwargs = {"horizon": 200,
        "offscreen_render": True, "camera_names": camera_name,
        "camera_heights": 512, "camera_widths": 512,
        "gripper_types": "Robotiq85TouchGripper",
        # "base_types": 'NullMount',
        "ignore_done": True}
        # "init_cube_pos": [0, 0]}

    robot_keys = ['robot0_joint_pos_cos', 'robot0_joint_pos_sin']

    if env_name == "TransferReach" or env_name == "TransferTrack":
        env_name = "Reach"
        taskname = "TransferReach"
        src_env_kwargs.update({"table_full_size": (0.6, 0.6, 0.05)})
        tgt_env_kwargs.update({"table_full_size": (0.6, 0.6, 0.05)})
        object_keys = ['target_pos']
    elif env_name == "TransferLift":
        env_name = "Lift"
        src_env_kwargs.update({"use_touch_obs": True, "table_offset": (0., 0., 0.908)})
        tgt_env_kwargs.update({"use_touch_obs": True, "table_offset": (0., 0., 0.908)})
        object_keys = [  
            'robot0_gripper_width',
            'cube_to_robot0_eef_pos',
            'cube_to_target_pos',
            'robot0_touch',
        ]
    elif env_name == "TransferPickPlace":
        env_name = "PickPlaceBread"
        src_env_kwargs.update({"use_touch_obs": True, 
            "bin1_pos": (-0.05, -0.25, 0.90), "bin2_pos": (-0.05, 0.28, 0.90)})
        tgt_env_kwargs.update({"use_touch_obs": True, 
            "bin1_pos": (-0.05, -0.25, 0.90), "bin2_pos": (-0.05, 0.28, 0.90)})
        object_keys = [  
            'robot0_gripper_width',
            'Bread_to_robot0_eef_pos',
            'Bread_to_Bread_bin_pos',
            'robot0_touch',
        ]
    keys = robot_keys + object_keys

    src_env = make_robosuite_env(env_name, "Panda", "JOINT_VELOCITY", **src_env_kwargs,
        render=render, initialization_noise=None)
    tgt_env = make_robosuite_env(env_name, "xArm6", "JOINT_VELOCITY", **tgt_env_kwargs,
        render=render, initialization_noise=None)


    obs = tgt_env.reset()
    print(np.allclose(obs['robot0_eef_pos'], np.array([-0.2, 0, 1.05]), atol=1e-3))

    src_obs = src_env.reset()
    tgt_obs = tgt_env.reset()
    src_obs_dim, tgt_obs_dim = {}, {}
    src_obs_dim["robot_dim"] = np.concatenate([src_obs[k] for k in robot_keys]).shape[0]
    tgt_obs_dim["robot_dim"] = np.concatenate([tgt_obs[k] for k in robot_keys]).shape[0]
    src_obs_dim["obj_dim"] = np.concatenate([src_obs[k] for k in object_keys]).shape[0]
    tgt_obs_dim["obj_dim"] = np.concatenate([tgt_obs[k] for k in object_keys]).shape[0]
    src_act_dim = src_env.robots[0].dof
    tgt_act_dim = tgt_env.robots[0].dof

    lat_obs_dim = 4
    lat_act_dim = 3

    src_obs_enc = build_mlp(src_obs_dim["robot_dim"], lat_obs_dim, 3, 256,
        activation='relu', output_activation='tanh').to(device)
    src_act_dec = build_mlp(lat_act_dim+src_obs_dim["robot_dim"], src_act_dim-1, 3, 256,
        activation='relu', output_activation='tanh').to(device)
    src_obs_enc.load_state_dict(torch.load(f'{policies_dir}/{taskname.lower()}_src_obs_enc.pt'))  
    src_act_dec.load_state_dict(torch.load(f'{policies_dir}/{taskname.lower()}_src_act_dec.pt'))

    tgt_obs_enc = build_mlp(tgt_obs_dim["robot_dim"], lat_obs_dim, 3, 256,
        activation='relu', output_activation='tanh').to(device)
    tgt_act_dec = build_mlp(lat_act_dim+tgt_obs_dim["robot_dim"], tgt_act_dim-1, 3, 256,
        activation='relu', output_activation='tanh').to(device)
    tgt_obs_enc.load_state_dict(torch.load(f'{policies_dir}/{taskname.lower()}_tgt_obs_enc.pt'))  
    tgt_act_dec.load_state_dict(torch.load(f'{policies_dir}/{taskname.lower()}_tgt_act_dec.pt'))

    lat_actor = ActorCustomInit(lat_obs_dim + src_obs_dim["obj_dim"], 
        lat_act_dim+1, 3, 256).to(device)
    lat_actor.load_state_dict(torch.load(f'{policies_dir}/{taskname.lower()}_lat_actor.pt'))

    return src_env, tgt_env, src_obs_enc, src_act_dec, lat_actor, tgt_obs_enc, tgt_act_dec


def test_task_transfer(task, evaluate_source=False, evaluate_target=False, evaluate_xarm=True, simulate_xarm=True,
                       save_obs=False, n_episodes=5, render=False, debug=False):
    assert task in ["TransferReach", "TransferTrack", "TransferLift", "TransferPickPlace"], \
        "Invalid task name, choose from: TransferReach, TransferTrack, TransferLift, TransferPickPlace"

    print(f"\n\nTASK: {task}\n")
    if task == "TransferReach":
        evaluate_xarm_func = partial(evaluate_xarm_reach, n_episodes=n_episodes, simulate_xarm=simulate_xarm, debug=debug, save_obs=save_obs)
    elif task == "TransferTrack":
        evaluate_xarm_func = partial(evaluate_xarm_reach, n_episodes=n_episodes, simulate_xarm=simulate_xarm, debug=debug, save_obs=save_obs,\
                                                          episodic=False, target_src="camera")
    elif task == "TransferLift":
        evaluate_xarm_func = partial(evaluate_xarm_lift, n_episodes=n_episodes, simulate_xarm=simulate_xarm, debug=debug, save_obs=save_obs)
    elif task == "TransferPickPlace":
        evaluate_xarm_func = partial(evaluate_xarm_pickplace, n_episodes=n_episodes, simulate_xarm=simulate_xarm, debug=debug, save_obs=save_obs)

    src_env, tgt_env, src_obs_enc, src_act_dec, lat_actor, tgt_obs_enc, tgt_act_dec = setup_transfer_envs(task)

    if evaluate_source:
        print("\nEvaluating latent policy on source robot")
        mean_reward = 0
        mean_dist = 0
        for _ in range(n_episodes):
            ep_reward, dist = evaluate_robosuite(task, src_env, src_obs_enc, src_act_dec, lat_actor, render=render, save_obs=save_obs)
            mean_reward += ep_reward
            if task == "TransferReach":
                mean_dist += dist
        print("")
        if task == "TransferReach":
            print(f"Mean distance to target: {mean_dist/n_episodes*1000:.3f} mm")
        print(f"Mean episode reward: {mean_reward/n_episodes}\n")
    src_env.close()

    if evaluate_target:
        print("\nEvaluating latent policy on target robot")
        mean_reward = 0
        mean_dist = 0
        for _ in range(n_episodes):
            ep_reward, dist = evaluate_robosuite(task, tgt_env, tgt_obs_enc, tgt_act_dec, lat_actor, render=render, save_obs=save_obs)
            mean_reward += ep_reward
            if task == "TransferReach":
                mean_dist += dist
        print("")
        if task == "TransferReach":
            print(f"Mean distance to target: {mean_dist/n_episodes*1000:.3f} mm")
        print(f"Mean episode reward: {mean_reward/n_episodes}\n")
    tgt_env.close()

    if evaluate_xarm:
        print("\nEvaluating latent policy on xArm")
        evaluate_xarm_func(lat_actor, tgt_obs_enc, tgt_act_dec)


if __name__ == '__main__':
    # test_task_transfer("TransferReach", evaluate_source=True, evaluate_target=True, evaluate_xarm=True, simulate_xarm=True, n_episodes=2, render=False, debug=False)
    # test_task_transfer("TransferLift", evaluate_source=True, evaluate_target=True, evaluate_xarm=False, simulate_xarm=False, n_episodes=2, render=True, debug=False)
    # test_task_transfer("TransferLift", evaluate_source=False, evaluate_target=False, evaluate_xarm=True, simulate_xarm=False, n_episodes=1, render=False, debug=False)
    # test_task_transfer("TransferPickPlace", evaluate_source=False, evaluate_target=True, evaluate_xarm=False, simulate_xarm=False, save_obs=True, n_episodes=1, render=True, debug=False)
    # test_task_transfer("TransferPickPlace", evaluate_source=False, evaluate_target=False, evaluate_xarm=True, simulate_xarm=False, save_obs=True, n_episodes=1, render=False, debug=False)
    test_task_transfer("TransferTrack", evaluate_source=False, evaluate_target=False, evaluate_xarm=True, simulate_xarm=True, 
                       save_obs=True, n_episodes=1, render=False, debug=True)