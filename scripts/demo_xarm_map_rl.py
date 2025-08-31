import torch
import numpy as np
import os
import time
import gymnasium as gym
import sys

# Add project root to Python path (ensure it takes precedence over similarly named packages elsewhere)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from envs.visual_env import VisualStateEnv
from ManiSkill.map_rl_from_example.model import Agent

@torch.no_grad()
def load_map_rl_policy(checkpoint_path, device="cuda"):
    """
    Loads the MAP-RL policy trained with ManiSkill.
    """
    # Mock action space
    action_space = gym.spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
    
    # Mock envs object for Agent initialization
    class MockEnv:
        def __init__(self):
            self.unwrapped = self
            self.single_action_space = action_space

    mock_envs = MockEnv()

    # Sample observation for model initialization, should be a dict of tensors
    sample_obs = {
        "rgb": torch.zeros((1, 224, 224, 3), dtype=torch.uint8, device=device),
        "state": torch.zeros((1, 13), dtype=torch.float32, device=device) # 6 (pos) + 6 (vel) + 1 (grasp)
    }

    # Initialize the Agent
    # These parameters are based on script_map_dino.sh and ppo_map.py defaults
    agent = Agent(
        envs=mock_envs,
        sample_obs=sample_obs,
        vision_encoder="dino",
        num_tasks=5, # from ppo_map.py model_ids default
        decoder=None,
        use_map=True,
        device=device,
        start_condition_map=False,
        use_local_fusion=False,
        use_rel_pos_in_fusion=False
    ).to(device)

    # Load the checkpoint
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        # The DINO backbone is not saved in the checkpoint, so strict=False is needed.
        agent.load_state_dict(state_dict, strict=False)
        print(f"Policy loaded from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Using randomly initialized policy.")

    agent.eval()
    return agent

def demo_xarm_map_rl(
    simulated=False,
    checkpoint_path="path/to/your/checkpoint.pt",
    print_info=True,
    max_steps=2000
):
    """
    Runs a deployment demo for the ManiSkill-trained MAP-RL policy on the xArm6.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    imshape = (224, 224)

    # Load the policy
    policy = load_map_rl_policy(checkpoint_path, device)

    @torch.no_grad()
    def sample_action(obs, target_obj_idx):
        # Prepare state tensor
        state_obs = np.concatenate([
            obs['robot0_joint_pos'],
            obs['robot0_joint_vel'],
            obs['is_grasped'],
        ])

        st = torch.from_numpy(state_obs).float().to(device).unsqueeze(0)

        # Prepare RGB tensor
        # The env returns rgb image as (H, W, C) numpy array
        rgb = torch.from_numpy(obs["rgb"]).to(device).unsqueeze(0)
        
        # NOTE: The policy is trained for a sequential picking task.
        # This demo simplifies by targeting a single object index.
        # For the full sequential task, logic to track success of the first pick
        # and switching target_obj_idx would be needed.
        env_target_obj_idx = torch.tensor([target_obj_idx], dtype=torch.long, device=device)

        # Get action from policy
        model_obs = {"state": st, "rgb": rgb}
        action = policy.get_action(model_obs, env_target_obj_idx=env_target_obj_idx, deterministic=True)
        return action.squeeze(0).cpu().numpy()

    # Environment setup
    # Camera settings to match simulation: crop to 480x480, then resize to 224x224
    cam_kwargs = dict(middle_crop_imsize=(480, 480))
    env = VisualStateEnv(mode=4, simulated=simulated, has_gripper=True, imsize=imshape, cam_kwargs=cam_kwargs)
    
    obs = env.reset()
    done = False
    
    # For sequential task, we can define the order of objects to pick
    # These indices correspond to the order in `model_ids` in ppo_map.py
    # e.g., 0: apple, 1: lemon, 2: orange
    target_obj_indices = [0, 1] 
    current_target_idx = target_obj_indices[0]
    
    print(f"Starting demo. Targeting object with index: {current_target_idx}")
    for t in range(max_steps):
        if not done:
            action = sample_action(obs, current_target_idx)
            
            # Scale down actions for smoother real-world execution
            # action[:6] *= 0.5
            import time
            time.sleep(3)
            action[:6] *= 0.5


            obs, rew, done, info = env.step(action)
            
            if print_info:
                print(f"Step {t+1}/{max_steps}, Reward: {rew:.4f}, Done: {done}")

            # NOTE: This is a placeholder for switching logic for the sequential task.
            # For a real deployment, you'd need a way to detect task completion
            # (e.g., by checking if the object is in the basket).
            # The current VisualStateEnv does not provide this success information.
            # Example switching logic:
            # if t == max_steps // 2:
            #    current_target_idx = target_obj_indices[1]
            #    print(f"Switching to target object with index: {current_target_idx}")

    env.reset()
    env.close()
    print("Demo finished.")
    return done

if __name__ == "__main__":
    simulated = False  # Set to False for real robot
    
    # !!! IMPORTANT !!!
    # Please provide the correct path to your trained model checkpoint.
    # The checkpoint should be from training with script_map_dino.sh.
    # For example: "runs/YCB_sequential_xarm6_ppo_map_dino_zero/ckpt_3001.pt"
    checkpoint_path = "/home/erl-xarm6/sunghwan_ws/erl_xArm/ManiSkill/checkpoints/ckpt_1341.pt"
    
    print_info = True
    max_steps = 500

    if not os.path.exists(checkpoint_path):
        print("="*50)
        print(f"ERROR: Checkpoint file not found at '{checkpoint_path}'")
        print("Please update the `checkpoint_path` variable in the `if __name__ == '__main__'` block.")
        print("="*50)
    else:
        demo_xarm_map_rl(
            simulated=simulated,
            checkpoint_path=checkpoint_path,
            print_info=print_info,
            max_steps=max_steps
        )