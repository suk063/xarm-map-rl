import os
import numpy as np
import matplotlib.pyplot as plt
from ruamel.yaml import YAML
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
config_dir = os.path.join(parent_dir, "configs")
load_dir = os.path.join(current_dir, "outputs/compare_sim_real")
fig_dir = os.path.join(load_dir, "figs")

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

def plot_comparison(title, start_idx, n_params, xs, ys, label_names=None):
    """
        Make an (`n_params`, 1) subplot of values from `ys`. Each subplot has `k` lines, one for each element of `ys`.\n
        `xs`: list of `k` numpy arrays with x axis values (time/index) for each line.\n
        `ys`: list of `k` numpy arrays with y axis values for each line.\n
        `label_names`: list of `k` strings with names for each line.\n
        `start_idx`: index of first parameter to plot.\n
        `n_params`: number of parameters to plot.\n
    """
    if label_names is None:
        label_names = [f"{i}" for i in range(len(ys))]
    fig, axs = plt.subplots(n_params, 1, figsize=(12, min(8, n_params*2)))
    if n_params == 1:
        axs = [axs]
    axs[0].set_title(title)

    num_lines = len(ys)
    # Get N-1 shades of blue, skipping the lightest
    blue_colors = plt.cm.Blues(np.linspace(0.4, 0.5, num_lines-1))
    colors = list(blue_colors) + ['red']

    for i in range(n_params):
        for j in range(num_lines):
            axs[i].plot(xs[j], ys[j][:, i+start_idx], label=label_names[j], color=colors[j])

    handles, labels = axs[0].get_legend_handles_labels()
    plt.subplots_adjust(right=0.85)
    fig.legend(handles, labels, loc='center right', borderaxespad=1.0, fontsize="small")

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    plt.savefig(fig_dir + f"/fig_{title}.png")

def compare_with_rbs_episodes(count):
    obs_rbs_eps = []
    time_rbs_eps = []
    actions_rbs_eps = []
    for i in range(count):
        obs_rbs_eps.append(np.load(load_dir + f"/obs_rbs_{task}_{i+1}.npz")['obs_list'])
        time_rbs_eps.append(np.load(load_dir + f"/obs_rbs_{task}_{i+1}.npz")['time_list'])
        actions_rbs_eps.append(np.load(load_dir + f"/obs_rbs_{task}_{i+1}.npz")['actions'])
    xs = [*time_rbs_eps, time_xarm]
    ys = [*obs_rbs_eps, obs_xarm]
    label_names = [f"robosuite_{i}" for i in range(count)] + ["xarm"]
    for i in range(len(param_counts)):
        print(f"Plotting {param_names[i]} with start index {sum(param_counts[:i])} and count {param_counts[i]}")
        plot_comparison(param_names[i], sum(param_counts[:i]), param_counts[i], xs, ys, label_names)

    # import ipdb; ipdb.set_trace()
    ys = [*actions_rbs_eps, actions_xarm]
    plot_comparison("Joint velocity actions", 0, 6, xs, ys, label_names)
    plot_comparison("Gripper actions", 6, 1, xs, ys, label_names)

def compare_gripper_actions_to_widths(count):
    obs_rbs_eps = []
    time_rbs_eps = []
    for i in range(count):
        gripper_widths = np.load(load_dir + f"/obs_rbs_{task}_{i+1}.npz")['obs_list'][:, 18]
        gripper_actions = np.load(load_dir + f"/obs_rbs_{task}_{i+1}.npz")['actions'][:, -1]
        obs_rbs_eps.append(np.concatenate([gripper_widths[:, None], gripper_actions[:, None]], axis=1))
        time_rbs_eps.append(np.load(load_dir + f"/obs_rbs_{task}_{i+1}.npz")['time_list'])
    label_names = [f"robosuite_{i}" for i in range(count)]
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
    for i in range(2):
        for j in range(len(obs_rbs_eps)):
            axs[i].plot(time_rbs_eps[j], obs_rbs_eps[j][:, i], label=label_names[j])
    axs[0].set_title("Gripper Width")
    axs[1].set_title("Gripper Action")
    plt.legend()

def compare_jv_actions_vs_obs(count):
    jv_comp_rbs_eps = []
    time_rbs_eps = []
    for i in range(count):
        jv_obs_rbs = np.load(load_dir + f"/obs_rbs_{task}_{i+1}.npz")['obs_list'][:, 12:18]
        jv_action_rbs = np.load(load_dir + f"/obs_rbs_{task}_{i+1}.npz")['actions'][:, :6]
        jv_comp_rbs_eps.append(jv_obs_rbs - jv_action_rbs)
        time_rbs_eps.append(np.load(load_dir + f"/obs_rbs_{task}_{i+1}.npz")['time_list'])
    
    # Handle off-by-1 for xarm observations
    jv_obs_xarm = obs_xarm[1:, 12:18]
    jv_action_xarm = actions_xarm[:-1, :6]
    jv_comp_xarm = jv_obs_xarm - jv_action_xarm

    xs = [*time_rbs_eps, time_xarm[:-1]]
    ys = [*jv_comp_rbs_eps, jv_comp_xarm]
    label_names = [f"robosuite_{i}" for i in range(count)] + ["xarm"]

    plot_comparison("Joint velocity: observed - desired", 0, 6, xs, ys, label_names)


def check_obj_and_eef_pos(count):
    obj_eef_pos_rbs_eps = []
    time_rbs_eps = []
    for i in range(count):
        obj_pos_rbs = np.load(load_dir + f"/obs_rbs_{task}_{i+1}.npz")['obj_pos_list']
        eef_pos_rbs = np.load(load_dir + f"/obs_rbs_{task}_{i+1}.npz")['eef_pos_list']
        obj_eef_pos_rbs_eps.append(np.concatenate([obj_pos_rbs, eef_pos_rbs, eef_pos_rbs-obj_pos_rbs], axis=1))
        time_rbs_eps.append(np.load(load_dir + f"/obs_rbs_{task}_{i+1}.npz")['time_list'])
    
    obj_pos_xarm = np.load(load_dir + f"/obs_xarm_{task}.npz")['obj_pos_list']
    eef_pos_xarm = np.load(load_dir + f"/obs_xarm_{task}.npz")['eef_pos_list']
    obj_eef_pos_xarm = np.concatenate([obj_pos_xarm, eef_pos_xarm, eef_pos_xarm-obj_pos_xarm], axis=1)

    xs = [*time_rbs_eps, time_xarm]
    ys = [*obj_eef_pos_rbs_eps, obj_eef_pos_xarm]
    label_names = [f"robosuite_{i}" for i in range(count)]
    plot_comparison(f"{obj} position", 0, 3, xs, ys, label_names)
    plot_comparison("End-effector position", 3, 3, xs, ys, label_names)
    plot_comparison(f"{obj}_to_eef: end-effector position - {obj} position", 6, 3, xs, ys, label_names)


def check_rbs_ep_count(task):
    """
        Check how many episodes of robosuite data are available for the given task.
    """
    count = 0
    while True:
        try:
            np.load(load_dir + f"/obs_rbs_{task}_{count+1}.npz")
            count += 1
        except FileNotFoundError:
            break
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="", help="Task to compare")
    parser.add_argument("--num-episodes", type=int, default=1, help="Number of episodes to compare")
    args = parser.parse_args()

    if args.task == "":
        yaml = YAML(typ='safe')
        params = yaml.load(open(config_dir + '/config.yaml'))
        task = params['task'].lower()
    else:
        task = args.task.lower()

    obs_rbs = np.load(load_dir + f"/obs_rbs_{task}_1.npz")['obs_list']
    time_rbs = np.load(load_dir + f"/obs_rbs_{task}_1.npz")['time_list']
    actions_rbs = np.load(load_dir + f"/obs_rbs_{task}_1.npz")['actions']

    obs_xarm = np.load(load_dir + f"/obs_xarm_{task}.npz")['obs_list']
    time_xarm = np.load(load_dir + f"/obs_xarm_{task}.npz")['time_list']
    actions_xarm = np.load(load_dir + f"/obs_xarm_{task}.npz")['actions']
    time_xarm = time_xarm - time_xarm[0]

    print(f"\nObservations for xarm and robosuite attempting {task.upper()} task:")
    print(f"\nTime step differences for xarm and robosuite:")
    print(f"XArm - Length:{time_xarm.shape}, Average Diff: {np.diff(time_xarm).mean()}")
    print(f"Robosuite - Length:{time_rbs.shape}, Average Diff: {np.diff(time_rbs).mean()}\n\n")

    if task == "lift":
        param_names = ["robot0_joint_pos_cos",
                        "robot0_joint_pos_sin",
                        "robot0_joint_vel",
                        "robot0_gripper_width",
                        "cube_to_robot0_eef_pos",
                        "cube_to_target_pos",
                        "robot0_touch"]
        param_counts = [6, 6, 6, 1, 3, 3, 2]
        obj = "cube"
    elif task == "lift_bc":
        param_names = ['robot0_joint_pos_cos', 
                        'robot0_joint_pos_sin', 
                        'robot0_joint_vel', 
                        'is_grasped', 
                        'cube_to_robot0_eef_pos',
                        'cube_to_target_pos']
        param_counts = [6, 6, 6, 1, 3, 3]
    elif task == "transferlift":
        param_names = ['robot0_joint_pos_cos', 
                       'robot0_joint_pos_sin',
                       'robot0_gripper_width',
                       'cube_to_robot0_eef_pos',
                       'cube_to_target_pos',
                       'robot0_touch']
        param_counts = [6, 6, 1, 3, 3, 2]
        obj = "cube"
    elif task == "pickplace":
        param_names = ["robot0_joint_pos_cos",
                        "robot0_joint_pos_sin",
                        "robot0_joint_vel",
                        "Bread_to_robot0_eef_pos",
                        "Bread_to_Bread_bin_pos",
                        "Bread_bin_to_robot0_eef_pos",
                        "robot0_touch"]
        param_counts = [6, 6, 6, 3, 3, 3, 2]
        obj = "Bread"
    elif task == "pickplace_bc":
        param_names = ['robot0_joint_pos_cos', 
                        'robot0_joint_pos_sin', 
                        'robot0_gripper_width', 
                        'Bread_to_robot0_eef_pos',
                        'Bread_to_Bread_bin_pos',
                        'robot0_touch']
        param_counts = [6, 6, 1, 3, 3, 2]    
    elif task == "transferpickplace":
        param_names = ["robot0_joint_pos_cos",
                        "robot0_joint_pos_sin",
                        'robot0_gripper_width',
                        "Bread_to_robot0_eef_pos",
                        "Bread_to_Bread_bin_pos",
                        "robot0_touch"]
        param_counts = [6, 6, 1, 3, 3, 2]
        obj = "Bread"
    elif task == "stack":
        param_names = ["robot0_joint_pos_cos",
                        "robot0_joint_pos_sin",
                        "robot0_joint_vel",
                        "robot0_eef_to_cubeA_pos",
                        "robot0_eef_to_cubeB_pos",
                        "cubeA_to_cubeB_pos",
                        "robot0_touch"]
        param_counts = [6, 6, 6, 3, 3, 3, 2]
        obj = "cubeA"
    elif task == "reachcanvisual" or task == "reachcanvisual_long" \
        or task == "liftcanvisual" or task == "liftcanvisual_long":
        param_names = ["robot0_joint_pos_cos",
                        "robot0_joint_pos_sin",
                        "robot0_joint_vel",
                        "robot0_eef_pos",
                        "robot0_eef_quat"]
        param_counts = [6, 6, 6, 3, 4]
        obj = "can"

    # for i in range(len(param_counts)):
    #     plot_comparison(param_names[i], sum(param_counts[:i]), param_counts[i], [time_xarm, time_rbs], [obs_xarm, obs_rbs], ["xarm", "robosuite"])

    # plot_comparison("cube_to_target_pos", 22, 3)

    count = check_rbs_ep_count(task)
    print(f"Found {count} episodes of robosuite data for task {task}")
    compare_with_rbs_episodes(count)
    # compare_gripper_actions_to_widths(4)
    # compare_jv_actions_vs_obs(4)
    # check_obj_and_eef_pos(4)

    plt.show()