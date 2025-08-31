from ruamel.yaml import YAML
from functools import partial 
import argparse
import os

from demo_xarm_reach import full_reach_demo
from demo_xarm_lift import demo_with_lift_irl, demo_with_lift_bc, demo_with_lift_ppo
from demo_xarm_pickplace import demo_with_pickplace_irl, demo_with_pickplace_bc
from demo_xarm_stack import demo_with_stack_irl
from demo_policy_transfer import test_task_transfer
from demo_robosuite import test_imitation_robosuite

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
config_dir = os.path.join(parent_dir, "configs")

def test_imitation_xarm():
    run_episode = None
    if task == 'Reach':
        # Random targets are generated automatically
        run_episode = partial(full_reach_demo, max_steps=200, target_src="random", episodic=True)
    elif task == 'Track':
        # Reach policy with target 15cm above cube
        run_episode = partial(full_reach_demo, max_steps=250, target_src="camera", episodic=False)
    elif task == 'Lift':
        # Place 40mm green cube at [0.56, 0, -0.095]
        run_episode = partial(demo_with_lift_irl, max_steps=200, use_pose=False, save_obs=params["save_obs"])
        if "bc" in imitation_policy_type:
            run_episode = partial(demo_with_lift_bc, max_steps=200, save_obs=params["save_obs"])
            if imitation_policy_type.endswith("mnskill"):
                run_episode = partial(demo_with_lift_bc, max_steps=200, save_obs=params["save_obs"], maniskill_policy=True)
        if "ppo" in imitation_policy_type:
            run_episode = partial(demo_with_lift_ppo, max_steps=200, save_obs=params["save_obs"])
    elif task == 'PickPlace':
        # Place 40mm cube in a 9.5 x 14.5 cm area around [0.6, -0.15, -0.067], and place a 9.5 x 14.5 cm bin at [0.6975, 0.2575, -0.112]
        run_episode = partial(demo_with_pickplace_irl, max_steps=250, use_pose=False, save_obs=params["save_obs"])
        if "bc" in imitation_policy_type:
            run_episode = partial(demo_with_pickplace_bc, max_steps=250, use_pose=False, save_obs=params["save_obs"])
    elif task == 'Stack':
        # Place one 40mm red and one 55mm green cube in a 16 x 16 cm area around [0.56, 0, -0.112]
        run_episode = partial(demo_with_stack_irl, max_steps=250, save_obs=params["save_obs"])

    num_episodes = params['num_episodes']
    success_count = 0
    for i in range(num_episodes):
        print(f"\n\nEXPERIMENT {i+1}/{num_episodes}")
        success = run_episode(simulated=params["simulated"], print_info=False)
        if success:
            success_count += 1
        else:
            print(f"Failed {task} task!")
    print(f"\n\n\nSUCCESS RATE: {success_count}/{num_episodes} = {100*success_count/num_episodes}%\n")


def run_experiment_config(config_filename="config.yaml"):
    global task, params, imitation_policy_type

    # Load config file
    yaml = YAML(typ='safe')
    params = yaml.load(open(config_dir + '/' + config_filename, 'r'))
    task = params['task']

    imitation_policy_type = ""
    if task.split('_')[-1] == "BC":
        imitation_policy_type = "bc"
        task = task.split('_')[0]
    if task == "Lift":
        imitation_policy_type += "_ppo"

    test_transfer = partial(test_task_transfer, task, evaluate_source=False, 
                            n_episodes=params['num_episodes'], 
                            render=params["render"], simulate_xarm=params["simulated"], 
                            debug=False, save_obs=params["save_obs"])

    if params["run_env"] == "xarm":
        if "transfer" in task.lower():
            # test_task_transfer(task, evaluate_source=True, evaluate_target=False, evaluate_xarm=False, n_episodes=1, render=True, save_obs=False)
            test_transfer(evaluate_target=False, evaluate_xarm=True)
        else:
            test_imitation_xarm()
    elif params["run_env"] == "robosuite":
        if "transfer" in task.lower():
            # test_task_transfer(task, evaluate_source=True, evaluate_target=False, evaluate_xarm=False, n_episodes=1, render=True, save_obs=False)
            test_transfer(evaluate_target=True, evaluate_xarm=False)
        else:
            test_imitation_robosuite(task, imitation_policy_type, render=params["render"], save_obs=params["save_obs"], num_episodes=params['num_episodes'])


if __name__ == "__main__":
    # Parse arguments to get config file name
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    run_experiment_config(args.config)
