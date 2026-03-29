import sys
import numpy as np 
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerPIDBicycle(Controller):
    def __init__(self, model, 
                 # TODO 4.1.3: Tune PID Gains
                 kp=16, 
                 ki=0.01, 
                 kd=9.5):
        self.path = None
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.acc_ep = 0
        self.last_ep = 0
        self.dt = model.dt
        self.current_idx = 0
    
    def set_path(self, path):
        super().set_path(path)
        self.acc_ep = 0
        self.last_ep = 0
        self.current_idx = 0
    
    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None
        
        # Extract State
        x, y, yaw = info["x"], info["y"], info["yaw"]

        # Check if reached end of track
        if self.current_idx >= len(self.path) - 5:
            return 0.0

        # Search Nearest Target Locally
        min_idx, min_dist = utils.search_nearest_local(self.path, (x,y), self.current_idx, lookahead=50)
        self.current_idx = min_idx
        
        # TODO 4.1.3: PID Control for Bicycle Kinematic Model
        target = self.path[min_idx]
        theta_target = np.rad2deg(np.arctan2(target[1] - y, target[0] - x))
        theta_err_pos = theta_target - yaw
        theta_err_pos = (theta_err_pos + 180) % 360 - 180
        err = min_dist * np.sin(np.deg2rad(theta_err_pos))
        
        self.acc_ep += err * self.dt
        
        yaw_ref = target[2]
        heading_err = yaw_ref - yaw
        heading_err = (heading_err + 180) % 360 - 180
        
        next_delta = self.kp * err + self.ki * self.acc_ep + self.kd * info["v"] * np.sin(np.deg2rad(heading_err))
        
        self.last_ep = err
        # [end] TODO 4.1.3
        return next_delta