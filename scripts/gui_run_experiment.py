import customtkinter as ctk
from tkinter import filedialog, messagebox
import os
import yaml
import subprocess
import signal

from run_experiment import run_experiment_config

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
config_dir = os.path.join(parent_dir, "configs")

# Set the scaling factor for high-resolution displays
ctk.set_appearance_mode("Dark")  # Could be "Dark" or "Light"
ctk.set_default_color_theme("blue")  # Set the color theme

# Define the main class for the application
class RobotConfigApp(ctk.CTk):
    def __init__(self, master=None):
        super().__init__(master) 
        self.title("Robot Experiments Interface")
        self.geometry("1400x610")
        self.tk.call('tk', 'scaling', 1.5)

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self.task_offsets = {
            "Reach": [0, 0, 0.05],
            "Track": [0, 0, 0],
            "Lift": [-0.02, 0.015, 0.065],
            "Lift_BC": [-0.02, 0.025, 0.065], 
            "Stack": [-0.025, 0, 0.0365],
            "Stack_BC": [-0.025, 0, 0.0365],
            "PickPlace": [-0.03, 0, 0.02],
            "PickPlace_BC": [-0.03, 0.02, 0.06],
            "TransferReach": [0, 0, 0],
            "TransferTrack": [0, 0, 0],
            "TransferLift": [0.025, 0, 0.058],
            "TransferPickPlace": [0.025, 0, 0.058]
        }
        
        self.non_functional_tasks = set(["Lift_BC", "Stack", "Stack_BC", "TransferReach"])
        
        self.create_widgets()

    def create_widgets(self):
        base_font = ("Arial", 18)
        header_font = ("Arial", 24)
        button_font = ("Arial", 16)
        main_button_font = ("Arial", 20)
        status_font = ("Arial", 20, "italic")
        entry_width = 200

        # Header
        self.header_label = ctk.CTkLabel(self, text="Robot Experiments Interface", font=header_font)
        self.header_label.grid(row=0, column=0, columnspan=2, pady=(15,10))

        # Dark mode switch
        self.dark_mode_switch = ctk.CTkSwitch(self, text="Dark Mode", font=button_font, switch_width=30, switch_height=15,
                                              command=lambda: ctk.set_appearance_mode("Dark") if self.dark_mode_switch.get() else ctk.set_appearance_mode("Light"))
        self.dark_mode_switch.grid(row=0, column=1, pady=10, padx=(0,10), sticky="e")
        self.dark_mode_switch.select()


        ## ---------------------------------------- Status Frame ---------------------------------------- ##

        self.status_frame = ctk.CTkFrame(self)
        self.status_frame.grid(row=1, column=0, columnspan=2, pady=(10, 5), padx=10, sticky="ew")
        self.status_frame.grid_columnconfigure(0, weight=1)
        
        self.status_label = ctk.CTkLabel(self.status_frame, text="Status: Ready", font=status_font)
        self.status_label.grid(row=0, column=0, pady=15)


        ## ---------------------------------------- Left Frame ---------------------------------------- ##

        self.left_frame = ctk.CTkFrame(self)
        self.left_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        
        # Configure left frame grid
        self.left_frame.grid_columnconfigure(0, weight=1)
        self.left_frame.grid_columnconfigure(1, weight=3)
        self.left_frame.grid_columnconfigure(2, weight=1)

        # Task Selection Dropdown
        self.task_label = ctk.CTkLabel(self.left_frame, text="Select Task", font=base_font, width=entry_width)
        self.task_label.grid(row=0, column=0, pady=(50,5), sticky="w")
        self.task_dropdown = ctk.CTkOptionMenu(self.left_frame, values=["Reach", "Track", "Lift", "Lift_BC", "PickPlace", "PickPlace_BC", "Stack", "Stack_BC", "TransferReach", "TransferTrack", "TransferLift", "TransferPickPlace"], font=base_font, command=self.update_task)
        self.task_dropdown.grid(row=0, column=1, pady=(50,5), padx=10, sticky="ew")
        self.task_dropdown.set("Lift")
        self.task = self.task_dropdown.get()

        # Run Environment Radio Buttons
        self.run_env_label = ctk.CTkLabel(self.left_frame, text="Run Environment", font=base_font, width=entry_width)
        self.run_env_label.grid(row=1, column=0, pady=(15,0), sticky="w")
        self.run_env_var = ctk.StringVar(value="robosuite")
        self.run_env_radio_xarm = ctk.CTkRadioButton(self.left_frame, text="xarm", variable=self.run_env_var, value="xarm", command=self.update_env, font=button_font, border_width_checked=10)
        self.run_env_radio_xarm.grid(row=1, column=1, pady=(15,0), sticky="w")
        self.run_env_radio_robosuite = ctk.CTkRadioButton(self.left_frame, text="robosuite", variable=self.run_env_var, value="robosuite", command=self.update_env, font=button_font, border_width_checked=10)
        self.run_env_radio_robosuite.grid(row=1, column=2, pady=(15,0), sticky="w")

        # Simulated Option Switch
        self.simulated_switch = ctk.CTkSwitch(self.left_frame, text="Use Simulated xArm", font=base_font)
        self.simulated_switch.grid(row=2, column=1, pady=5, sticky="w")
        self.simulated_switch.select()
        self.simulated_switch.configure(state="disabled")

        # Render Option Switch
        self.render_switch = ctk.CTkSwitch(self.left_frame, text="Visualize Robosuite", font=base_font)
        self.render_switch.grid(row=2, column=2, pady=5, padx=(0, 10), sticky="w")
        self.render_switch.select()

        # Save Observations Checkbox
        self.save_obs_label = ctk.CTkLabel(self.left_frame, text="Save Observations", font=base_font, width=entry_width)
        self.save_obs_label.grid(row=3, column=0, pady=(15,0), sticky="w")
        self.save_obs_check = ctk.CTkCheckBox(self.left_frame, text="", font=base_font)
        self.save_obs_check.grid(row=3, column=1, pady=(15,0), padx=10, sticky="w")

        # Number of Episodes Input
        self.episodes_label = ctk.CTkLabel(self.left_frame, text="Number of Episodes", font=base_font, width=entry_width)
        self.episodes_label.grid(row=4, column=0, pady=(15,0), sticky="w")
        self.episodes_input = ctk.CTkEntry(self.left_frame, font=base_font, width=entry_width)
        self.episodes_input.grid(row=4, column=1, pady=(15,0), padx=10, sticky="ew")
        self.episodes_input.insert(0, "1")

        # Camera Cube Offset Input Fields
        self.offset_label = ctk.CTkLabel(self.left_frame, text="Camera Cube Offset", font=base_font, width=entry_width)
        self.offset_label.grid(row=5, column=0, pady=(15,0), sticky="w")
        self.offset_x_input = ctk.CTkEntry(self.left_frame, font=base_font)
        self.offset_y_input = ctk.CTkEntry(self.left_frame, font=base_font)
        self.offset_z_input = ctk.CTkEntry(self.left_frame, font=base_font)
        self.offset_x_input.grid(row=5, column=1, pady=(15,0), padx=10, sticky="ew")
        self.offset_y_input.grid(row=6, column=1, pady=(5,0), padx=10, sticky="ew")
        self.offset_z_input.grid(row=7, column=1, pady=5, padx=10, sticky="ew")

        # Model Path Text Input
        self.model_path_label = ctk.CTkLabel(self.left_frame, text="Model Path", font=base_font, width=entry_width)
        self.model_path_label.grid(row=8, column=0, pady=(15,0), sticky="w")
        self.model_path_input = ctk.CTkEntry(self.left_frame, font=base_font)
        self.model_path_input.grid(row=8, column=1, pady=(15,0), padx=10, sticky="ew")
        self.model_path_input.insert(0, "/home/erl-tianyu/xArm_robosuite/printed_cube_model.ply")
        self.model_path_button = ctk.CTkButton(self.left_frame, text="Browse", command=self.browse_file, font=button_font)
        self.model_path_button.grid(row=8, column=2, pady=(15,5), padx=10)


        ## ---------------------------------------- Right Frame ---------------------------------------- ##
        
        self.right_frame = ctk.CTkFrame(self)
        self.right_frame.grid(row=2, column=1, padx=10, pady=10, sticky="nsew")
        
        ## ----------------- Experiments Frame ----------------- ##
        self.experiments_frame = ctk.CTkFrame(self.right_frame)
        self.experiments_frame.pack(fill="x", padx=10, pady=10)
        self.experiments_frame.grid_columnconfigure(0, weight=1)  # Empty left column
        self.experiments_frame.grid_columnconfigure((1, 2, 3), weight=0)  # Content columns
        self.experiments_frame.grid_columnconfigure(4, weight=1)  # Empty right column

        # Save Button
        self.save_button = ctk.CTkButton(self.experiments_frame, text="Save Configuration", command=self.save_config, font=button_font)
        self.save_button.grid(row=1, column=1, pady=(15,5), padx=10, sticky="ew")

        # Load config_gui Button
        self.load_button = ctk.CTkButton(self.experiments_frame, text="Load config_gui", command=lambda: self.load_config("config_gui.yaml"), font=button_font)
        self.load_button.grid(row=1, column=2, pady=(15,5), padx=10, sticky="ew")

        # Browse and load config Button
        self.load_button = ctk.CTkButton(self.experiments_frame, text="Browse & Load Config", command=lambda: self.load_config(None), font=button_font)
        self.load_button.grid(row=1, column=3, pady=(15,5), padx=10, sticky="ew")

        # Compare xarm and robosuite Button
        self.compare_button = ctk.CTkButton(self.experiments_frame, text="Compare xarm vs robosuite", command=self.compare, font=button_font)
        self.compare_button.grid(row=2, column=2, pady=(5,0), padx=10, sticky="ew")

        # Run Experiment Button
        self.run_button = ctk.CTkButton(self.experiments_frame, text="Run Experiment", fg_color="green", command=self.run_experiment, font=main_button_font)
        self.run_button.grid(row=3, column=1, columnspan=3, pady=(15,0), sticky="ew")

        # Stop Experiment Button
        self.stop_button = ctk.CTkButton(self.experiments_frame, text="Stop Experiment", fg_color="red", command=self.stop_experiment, font=main_button_font)
        self.stop_button.grid(row=4, column=1, columnspan=3, pady=(5,15), sticky="ew")


        ## ----------------- Testing Tools Frame ----------------- ##
        self.testing_tools_frame = ctk.CTkFrame(self.right_frame)
        self.testing_tools_frame.pack(fill="x", padx=10, pady=10)

        # Test Camera Button
        self.test_camera_button = ctk.CTkButton(self.testing_tools_frame, text="Test Camera", fg_color="green", command=self.test_camera, font=button_font)
        self.test_camera_button.grid(row=0, column=1, pady=(15,5), padx=10)

        # Test FSR Button
        self.test_fsr_button = ctk.CTkButton(self.testing_tools_frame, text="Test FSR", fg_color="green", command=self.test_fsr, font=button_font)
        self.test_fsr_button.grid(row=0, column=2, pady=(15,5), padx=10)

        # Run VR Controller Test Button
        self.run_vr_test_button = ctk.CTkButton(self.testing_tools_frame, text="Test VR wand", fg_color="green", command=self.run_vr_test, font=button_font)
        self.run_vr_test_button.grid(row=0, column=3, pady=(15,5), padx=10)

        # Run Extrinsics Calibration Button
        self.run_extrinsics_button = ctk.CTkButton(self.testing_tools_frame, text="Run Extrinsics Calibration", fg_color="green", command=self.run_extrinsics_calibration, font=button_font)
        self.run_extrinsics_button.grid(row=0, column=4, pady=(15,5), padx=10)

        # Stop Test Button
        self.stop_test_button = ctk.CTkButton(self.testing_tools_frame, text="Stop Test", fg_color="red", command=self.stop_test, font=main_button_font)
        self.stop_test_button.grid(row=1, column=1, columnspan=4, pady=(0,15), padx=10, sticky="ew")
        # self.stop_test_button.configure(state="disabled")


        ## ----------------- Teleoperation Frame ----------------- ##
        self.teleop_frame = ctk.CTkFrame(self.right_frame)
        self.teleop_frame.pack(fill="x", padx=10, pady=10)
        self.teleop_frame.grid_columnconfigure(0, weight=1)  # Empty left column
        self.teleop_frame.grid_columnconfigure((1, 2, 3, 4), weight=0)  # Content columns
        self.teleop_frame.grid_columnconfigure(5, weight=1)  # Empty right column

        # Teleoperation Active EEF Axes
        self.teleop_active_eef_axes_label = ctk.CTkLabel(self.teleop_frame, text="Teleop: Active EEF Axes", font=base_font, width=entry_width)
        self.teleop_active_eef_axes_label.grid(row=0, column=1, pady=10, sticky="w")

        self.teleop_active_eef_axes_r = ctk.CTkCheckBox(self.teleop_frame, text="roll", font=base_font)
        self.teleop_active_eef_axes_r.grid(row=0, column=2, pady=15, sticky="ew")

        self.teleop_active_eef_axes_p = ctk.CTkCheckBox(self.teleop_frame, text="pitch", font=base_font)
        self.teleop_active_eef_axes_p.grid(row=0, column=3, pady=15, sticky="ew")

        self.teleop_active_eef_axes_y = ctk.CTkCheckBox(self.teleop_frame, text="yaw", font=base_font)
        self.teleop_active_eef_axes_y.grid(row=0, column=4, pady=15, sticky="ew")
        self.teleop_active_eef_axes_y.select()

        self.teleop_run_button = ctk.CTkButton(self.teleop_frame, text="Run Teleoperation", fg_color="green", command=self.run_teleoperation, font=main_button_font)
        self.teleop_run_button.grid(row=1, column=1, columnspan=4, pady=(5,0), padx=10, sticky="ew")

        self.teleop_stop_button = ctk.CTkButton(self.teleop_frame, text="Stop Teleoperation", fg_color="red", command=self.stop_teleoperation, font=main_button_font)
        self.teleop_stop_button.grid(row=2, column=1, columnspan=4, pady=(5,15), padx=10, sticky="ew")


        # Initialize offsets
        self.update_offsets()

        self.process = None
        self.test_process = None
        self.teleop_process = None

    def update_task(self, task=None):
        self.task = task
        if task in self.non_functional_tasks:
            messagebox.showwarning("Warning", f"The '{task}' task may not work as expected.")
        if task == "Track" or task == "TransferTrack":
            self.run_env_var.set("xarm")
            self.update_env()
            self.run_env_radio_robosuite.configure(state="disabled")
        elif self.run_env_var.get() == "robosuite":
            self.run_env_radio_robosuite.configure(state="normal")

        self.update_offsets()

    def update_offsets(self):
        # Update offset fields based on selected task
        offsets = self.task_offsets[self.task]
        self.offset_x_input.delete(0, ctk.END)
        self.offset_x_input.insert(0, str(offsets[0]))
        self.offset_y_input.delete(0, ctk.END)
        self.offset_y_input.insert(0, str(offsets[1]))
        self.offset_z_input.delete(0, ctk.END)
        self.offset_z_input.insert(0, str(offsets[2]))

    def update_env(self):
        # Update environment-specific options
        if self.run_env_var.get() == "xarm":
            self.simulated_switch.configure(state="normal")
            self.render_switch.configure(state="disabled")
        else:
            self.simulated_switch.configure(state="disabled")
            self.render_switch.configure(state="normal")

    def browse_file(self):
        # Open file dialog and set model path
        file_path = filedialog.askopenfilename(filetypes=[("PLY files", "*.ply")])
        if file_path:
            self.model_path_input.delete(0, ctk.END)
            self.model_path_input.insert(0, file_path)

    def save_config(self):
        self.task_offsets[self.task_dropdown.get()] = [float(self.offset_x_input.get()), float(self.offset_y_input.get()), float(self.offset_z_input.get())]

        # Save current configuration to a file
        config = {
            "task": self.task_dropdown.get(),
            "run_env": self.run_env_var.get(),
            "simulated": bool(self.simulated_switch.get()),
            "render": bool(self.render_switch.get()),
            "save_obs": bool(self.save_obs_check.get()),
            "num_episodes": int(self.episodes_input.get()),
            "camera_cube_offset": self.task_offsets,
            "model_path": self.model_path_input.get(),
            "teleoperation": {"active_eef_axes": [self.teleop_active_eef_axes_r.get(), 
                                                  self.teleop_active_eef_axes_p.get(), 
                                                  self.teleop_active_eef_axes_y.get()]}
        }
        config_filename = config_dir + "/config_gui.yaml"
        if config_filename:
            with open(config_filename, "w") as f:
                yaml.dump(config, f)
            self.status_label.configure(text=f"Status: Saved configuration to {os.path.basename(config_filename)}")

    def load_config(self, config_filename):
        # Load configuration from a file
        if config_filename == None:
            try:
                config_filename = filedialog.askopenfilename(initialdir=config_dir, filetypes=[("YAML files", "*.yaml")])
            except FileNotFoundError:
                self.status_label.configure(text=f"Status: No config file selected.")
                return
        config_filepath = f"{config_dir}/{config_filename}"
        if config_filename:
            config = yaml.load(open(config_filepath, 'r'), Loader=yaml.FullLoader)
            self.task_dropdown.set(config["task"])
            self.task = config["task"]
            self.run_env_var.set(config["run_env"])
            self.simulated_switch.setvar("ctkValue", config["simulated"])
            self.render_switch.setvar("ctkValue", config["render"])
            self.save_obs_check.setvar("ctkValue", config["save_obs"])
            self.episodes_input.delete(0, ctk.END)
            self.episodes_input.insert(0, config["num_episodes"])
            self.offset_x_input.delete(0, ctk.END)
            self.model_path_input.delete(0, ctk.END)
            self.model_path_input.insert(0, config["model_path"])
            self.task_offsets = config["camera_cube_offset"]
            self.update_offsets()
            self.status_label.configure(text=f"Status: Loaded configuration from {os.path.basename(config_filename)}")

    def compare(self):
        # Compare xarm and robosuite
        self.save_config()
        self.status_label.configure(text=f"Status: Comparing xarm vs robosuite for {self.task}...")
        
        # Check if a previous subprocess exists and is running
        if self.process and not self.process.poll():
            self.status_label.configure(text="Status: Experiment is already running.")
            return

        print("\n\n")
        # Run using subprocess to avoid Tkinter freezing and to allow killing the process
        self.process = subprocess.Popen(["python", current_dir + "/compare_xarm_robosuite.py", "--task", self.task, "--num-episodes", self.episodes_input.get()])
        self.stop_button.configure(state="normal")

    # Test Camera Command Function
    def test_camera(self):
        self.status_label.configure(text=f"Status: Running camera test...")
        self.run_test_subprocess(os.path.join(parent_dir, "devices", "camera.py"))

    # Test FSR Command Function
    def test_fsr(self):
        self.status_label.configure(text=f"Status: Running FSR test...")
        self.run_test_subprocess(os.path.join(parent_dir, "devices", "fsr.py"))

    # Run Extrinsics Calibration Command Function
    def run_extrinsics_calibration(self):
        self.status_label.configure(text=f"Status: Running camera extrinsics calibration...")
        self.run_test_subprocess(os.path.join(current_dir, "camera_extrinsics_calibration.py"))

    def run_vr_test(self):
        self.status_label.configure(text=f"Status: Running VR Controller test...")
        self.run_test_subprocess(os.path.join(parent_dir, "devices", "vr_controller.py"))

    # Run Subprocess
    def run_test_subprocess(self, script_path):
        if self.test_process is not None and self.test_process.poll() is None:
            self.status_label.configure(text="Status: A test is already running.")
            return
        print("\n\n")
        self.test_process = subprocess.Popen(["python", script_path])
        self.stop_test_button.configure(state="normal")

    # Stop Test Command Function
    def stop_test(self):
        if self.test_process:
            self.test_process.send_signal(signal.SIGINT)
            self.test_process = None
            # self.stop_test_button.configure(state="disabled")
            self.status_label.configure(text="Status: Test stopped")

    def run_experiment(self):
        # Run the experiment with current configuration
        self.save_config()
        self.status_label.configure(text=f"Status: Running {self.task} experiment...")
        
        # Check if a previous subprocess exists and is running
        if self.process is not None and self.process.poll() is None:
            self.status_label.configure(text="Status: Experiment is already running.")
            return

        print("\n\n")
        # Run using subprocess to avoid Tkinter freezing and to allow killing the process
        self.process = subprocess.Popen(["python", current_dir + "/run_experiment.py", "--config", "config_gui.yaml"])
        self.stop_button.configure(state="normal")

    def stop_experiment(self):
        if self.process and not self.process.poll():
            self.process.send_signal(signal.SIGINT)
            self.process = None
            # self.stop_button.configure(state="disabled")
            self.status_label.configure(text="Status: Experiment stopped")

    def run_teleoperation(self):
        if self.run_env_var.get() == "robosuite":
            self.run_env_var.set("xarm")
            self.update_env()
            self.simulated_switch.select()

        self.save_config()

        sim_mode = "simulated" if self.simulated_switch.get() else "real"
        self.status_label.configure(text=f"Status: Running teleoperation on {sim_mode} xarm...")
        
        # Check if a previous subprocess exists and is running
        if self.teleop_process is not None and self.teleop_process.poll() is None:
            self.status_label.configure(text="Status: Teleoperation is already running.")
            return

        print("\n\n")
        # Run using subprocess to avoid Tkinter freezing and to allow killing the process
        self.teleop_process = subprocess.Popen(["python", current_dir + "/robot_teleoperation.py", "--config", "config_gui.yaml"])

    def stop_teleoperation(self):
        if self.teleop_process and not self.teleop_process.poll():
            self.teleop_process.send_signal(signal.SIGINT)
            self.teleop_process = None
            # self.stop_button.configure(state="disabled")
            self.status_label.configure(text="Status: Teleoperation stopped")


# Run the application
if __name__ == "__main__":
    app = RobotConfigApp()
    app.mainloop()
