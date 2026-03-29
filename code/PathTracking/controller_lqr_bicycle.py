import sys
import numpy as np 
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerLQRBicycle(Controller):
    def __init__(self, model, Q=None, R=None, control_state='steering_angle'):
        self.path = None
        if control_state == 'steering_angle':
            self.Q = np.eye(2)
            self.R = np.eye(1)
            # TODO 4.4.1: Tune LQR Gains
            self.Q[0,0] = 50.0  # 強烈懲罰橫向誤差
            self.Q[1,1] = 1.0   # 稍微懲罰航向誤差
            self.R[0,0] = 15.0   # 轉方向盤的代價
        elif control_state == 'steering_angular_velocity':
            self.Q = np.eye(3)
            self.R = np.eye(1)
            # TODO 4.4.4: Tune LQR Gains
            self.Q[0,0] = 100.0 # 強烈懲罰橫向誤差
            self.Q[1,1] = 5.0   # 懲罰航向誤差
            self.Q[2,2] = 0.1   # 懲罰方向盤角度(希望方向盤盡量回正)
            self.R[0,0] = 15.0   # 轉方向盤角速度的代價
        self.pe = 0
        self.pth_e = 0
        self.pdelta = 0
        self.dt = model.dt
        self.l = model.l
        self.control_state = control_state
        self.current_idx = 0

    def set_path(self, path):
        super().set_path(path)
        self.pe = 0
        self.pth_e = 0
        self.pdelta = 0
        self.current_idx = 0

    def _solve_DARE(self, A, B, Q, R, max_iter=150, eps=0.01): # Discrete-time Algebra Riccati Equation (DARE)
        P = Q.copy()
        for i in range(max_iter):
            temp = np.linalg.inv(R + B.T @ P @ B)
            Pn = A.T @ P @ A - A.T @ P @ B @ temp @ B.T @ P @ A + Q
            if np.abs(Pn - P).max() < eps:
                break
            P = Pn
        return Pn

    # State: [x, y, yaw, delta, v]
    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None
        
        # Extract State 
        x, y, yaw, delta, v = info["x"], info["y"], info["yaw"], info["delta"], info["v"]
        yaw = utils.angle_norm(yaw)

        # Check if reached end of track
        if self.current_idx >= len(self.path) - 3:
            return 0.0
        
        # Search Nesrest Target
        min_idx, min_dist = utils.search_nearest(self.path, (x,y))
        target = self.path[min_idx]
        target[2] = utils.angle_norm(target[2])
        
        if self.control_state == 'steering_angle':
            # TODO 4.4.1: LQR Control for Bicycle Kinematic Model with steering angle as control input
            yaw_ref = target[2]
      
            dx = x - target[0]
            dy = y - target[1]
            nx = -np.sin(np.deg2rad(yaw_ref))
            ny = np.cos(np.deg2rad(yaw_ref))
            e = dx * nx + dy * ny 
  
            theta_e = yaw - yaw_ref 
            theta_e = (theta_e + 180) % 360 - 180
            theta_e_rad = np.deg2rad(theta_e)
  
            x_state = np.zeros((2, 1))
            x_state[0, 0] = e
            x_state[1, 0] = theta_e_rad

            v_safe = max(v, 0.1)
            
            A = np.zeros((2, 2))
            A[0, 0] = 1.0
            A[0, 1] = v_safe * self.dt
            A[1, 0] = 0.0
            A[1, 1] = 1.0
            
            B = np.zeros((2, 1))
            B[0, 0] = 0.0
            B[1, 0] = (v_safe * self.dt) / self.l

            P = self._solve_DARE(A, B, self.Q, self.R)
            K = np.linalg.inv(self.R + B.T @ P @ B) @ (B.T @ P @ A)

            u = -K @ x_state
            next_delta = np.rad2deg(u[0, 0])
            next_delta = np.clip(next_delta, -40.0, 40.0)
            
            self.pe = e
            self.pth_e = theta_e_rad
            # [end] TODO 4.4.1
        elif self.control_state == 'steering_angular_velocity':
            # TODO 4.4.4: LQR Control for Bicycle Kinematic Model with steering angular velocity as control input
            # TODO 4.4.4: LQR Control for Bicycle Kinematic Model with steering angular velocity as control input
 
            yaw_ref = target[2]
            
            dx = x - target[0]
            dy = y - target[1]
            nx = -np.sin(np.deg2rad(yaw_ref))
            ny = np.cos(np.deg2rad(yaw_ref))
            e = dx * nx + dy * ny 
            
            theta_e = yaw - yaw_ref 
            theta_e = (theta_e + 180) % 360 - 180
            theta_e_rad = np.deg2rad(theta_e)

            delta_rad = np.deg2rad(delta)
            
            x_state = np.zeros((3, 1))
            x_state[0, 0] = e
            x_state[1, 0] = theta_e_rad
            x_state[2, 0] = delta_rad
            
            v_safe = max(v, 0.1)
            
            A = np.zeros((3, 3))
            A[0, 0] = 1.0
            A[0, 1] = v_safe * self.dt
            A[0, 2] = 0.0
            
            A[1, 0] = 0.0
            A[1, 1] = 1.0
            A[1, 2] = (v_safe * self.dt) / self.l
            
            A[2, 0] = 0.0
            A[2, 1] = 0.0
            A[2, 2] = 1.0
            
            B = np.zeros((3, 1))
            B[0, 0] = 0.0
            B[1, 0] = 0.0
            B[2, 0] = self.dt
            
            P = self._solve_DARE(A, B, self.Q, self.R)
            K = np.linalg.inv(self.R + B.T @ P @ B) @ (B.T @ P @ A)
         
            u_omega = -K @ x_state
       
            next_delta_rad = delta_rad + u_omega[0, 0] * self.dt
            next_delta = np.rad2deg(next_delta_rad)
            next_delta = np.clip(next_delta, -40.0, 40.0)
            
            self.pe = e
            self.pth_e = theta_e_rad
            self.pdelta = delta_rad
            # [end] TODO 4.4.4
        
        return next_delta
