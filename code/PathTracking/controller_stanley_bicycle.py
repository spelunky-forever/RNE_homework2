import sys
import numpy as np 
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerStanleyBicycle(Controller):
    def __init__(self, model, 
                 # TODO 4.3.1: Tune Stanley Gain
                 kp=1):
        self.path = None
        self.kp = kp
        self.l = model.l
        self.current_idx = 0

    def set_path(self, path):
        super().set_path(path)
        self.current_idx = 0

    # State: [x, y, yaw, delta, v]
    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None
        
        # Extract State 
        x, y, yaw, delta, v = info["x"], info["y"], info["yaw"], info["delta"], info["v"]

        # Check if reached end of track
        if self.current_idx >= len(self.path) - 5:
            return 0.0

        # Search Front Wheel Target Locally
        front_x = x + self.l*np.cos(np.deg2rad(yaw))
        front_y = y + self.l*np.sin(np.deg2rad(yaw))
        vf = v / np.cos(np.deg2rad(delta)) if np.cos(np.deg2rad(delta)) != 0 else v
        
        min_idx, min_dist = utils.search_nearest_local(self.path, (front_x,front_y), self.current_idx, lookahead=50)
        self.current_idx = min_idx
        target = self.path[min_idx]

        # TODO 4.3.1: Stanley Control for Bicycle Kinematic Model
        yaw_ref = target[2]
        theta_e = yaw_ref - yaw
        theta_e = (theta_e + 180) % 360 - 180
        dx = front_x - target[0]
        dy = front_y - target[1]
        nx = -np.sin(np.deg2rad(yaw_ref))
        ny = np.cos(np.deg2rad(yaw_ref))
        e = dx * nx + dy * ny
        epsilon = 1e-6
        if vf < epsilon:
            vf = epsilon
        theta_e_rad = np.deg2rad(theta_e)
        stanley_rad = theta_e_rad + np.arctan( (-self.kp * e) / vf )
        next_delta = np.rad2deg(stanley_rad)
        # [end] TODO 4.3.1
    
        return next_delta
