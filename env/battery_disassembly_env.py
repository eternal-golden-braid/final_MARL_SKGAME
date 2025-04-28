"""
Battery Disassembly Environment for Multi-Robot Coordination.

This module provides an environment for three robots to coordinate on
disassembling a battery module with different types of tasks.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, Tuple, List, Optional, Union, Any


class BatteryDisassemblyEnv:
    """
    Environment class for the battery disassembly task with three robots.
    
    This environment models a workstation with a battery module and three robots:
    - Franka robot (Leader): Equipped with a two-finger gripper for unbolting operations
    - UR10 robot (Follower 1): Equipped with vacuum suction for sorting and pick-and-place
    - Kuka robot (Follower 2): Equipped with specialized tools for casing and connections
    
    Attributes:
        rng (np.random.Generator): Random number generator
        task_id (int): ID of the current task
        task_board (np.ndarray): Board representing tasks to be completed
        task_prop (Dict): Properties of each task
        curr_board (np.ndarray): Current state of the task board
        franka_pos (np.ndarray): Position of the Franka robot
        ur10_pos (np.ndarray): Position of the UR10 robot
        kuka_pos (np.ndarray): Position of the Kuka robot
        battery_pos (np.ndarray): Position of the battery module
        bin_positions (Dict): Positions of the bins
        completed_tasks (List): List of completed tasks
        franka_state (Dict): State of the Franka robot
        ur10_state (Dict): State of the UR10 robot
        kuka_state (Dict): State of the Kuka robot
        time_step (int): Current time step
        max_time_steps (int): Maximum number of time steps
        franka_workspace_radius (float): Workspace radius of the Franka robot
        ur10_workspace_radius (float): Workspace radius of the UR10 robot
        kuka_workspace_radius (float): Workspace radius of the Kuka robot
        franka_failure_prob (float): Failure probability of the Franka robot
        ur10_failure_prob (float): Failure probability of the UR10 robot
        kuka_failure_prob (float): Failure probability of the Kuka robot
        debug (bool): Whether to print debug information
    """

    def __init__(self, parameters: Dict[str, Any]):
        """
        Initialize the battery disassembly environment.
        
        Args:
            parameters: Dictionary containing environment parameters
                - seed: Random seed
                - task_id: ID of the task
                - max_time_steps: Maximum number of time steps
                - franka_failure_prob: Failure probability of the Franka robot
                - ur10_failure_prob: Failure probability of the UR10 robot
                - kuka_failure_prob: Failure probability of the Kuka robot
                - debug: Whether to print debug information
        """
        self.rng = np.random.default_rng(parameters['seed'])
        self.task_id = parameters['task_id']
        self.debug = parameters.get('debug', False)
        
        # Load the task board and properties
        self.task_board, self.task_prop = self._task_reader(self.task_id)
        self.curr_board = np.copy(self.task_board)
        
        # Define robot properties
        self.franka_pos = np.array([0.5, -0.3, 0.5])   # Base position of Franka robot (Leader)
        self.ur10_pos = np.array([-0.5, -0.3, 0.5])    # Base position of UR10 robot (Follower 1)
        self.kuka_pos = np.array([0.0, -0.5, 0.5])     # Base position of Kuka robot (Follower 2)
        
        # Define workspace properties
        self.battery_pos = np.array([0.0, 0.0, 0.1])  # Position of the battery module
        self.bin_positions = {
            'screws': np.array([0.3, 0.4, 0.1]),
            'cells': np.array([-0.3, 0.4, 0.1]),
            'casings': np.array([0.0, 0.5, 0.1]),
            'connectors': np.array([0.3, -0.4, 0.1])
        }
        
        # Task completion tracking
        self.completed_tasks = []
        
        # Robot states
        self.franka_state = {'position': self.franka_pos, 'gripper_open': True, 'holding': None}
        self.ur10_state = {'position': self.ur10_pos, 'suction_active': False, 'holding': None}
        self.kuka_state = {'position': self.kuka_pos, 'tool_active': False, 'holding': None}
        
        # Task timing and resource tracking
        self.time_step = 0
        self.max_time_steps = parameters.get('max_time_steps', 100)
        
        # Robot kinematic constraints
        self.franka_workspace_radius = 0.8
        self.ur10_workspace_radius = 1.0
        self.kuka_workspace_radius = 0.9
        
        # Task failure probabilities (uncertainty modeling)
        self.franka_failure_prob = parameters.get('franka_failure_prob', 0.1)
        self.ur10_failure_prob = parameters.get('ur10_failure_prob', 0.1)
        self.kuka_failure_prob = parameters.get('kuka_failure_prob', 0.1)
        
        # Reward shaping parameters
        self.reward_step_penalty = parameters.get('reward_step_penalty', 0.1)
        self.reward_task_cleared = parameters.get('reward_task_cleared', 1.0)
        self.reward_row_bonus = parameters.get('reward_row_bonus', 2.0)
        self.reward_collision_penalty = parameters.get('reward_collision_penalty', 1.0)
        
    def _task_reader(self, task_id: int) -> Tuple[np.ndarray, Dict]:
        """
        Read the task information from the configuration files.
        Extended for three-robot scenario with more task types.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Tuple containing the task board and task properties
        """
        # Task board represents the spatial arrangement of components to be disassembled
        # 0: Empty space
        # 1-4: Top screws (requires unbolting by Franka)
        # 5-8: Side screws (requires unbolting by Franka)
        # 9-12: Battery cells (requires pick-and-place by UR10)
        # 13-16: Casing components (requires specialized tools by Kuka)
        # 17-20: Connectors (requires collaborative effort between UR10 and Kuka)
        # 21-22: Complex assemblies (requires all three robots)
        
        # For task_id 1, use the default task board
        if task_id == 1:
            task_board = np.array([
                [1, 2, 3, 4],
                [9, 10, 11, 12],
                [17, 18, 19, 20],
                [5, 6, 7, 8],
                [13, 14, 15, 16],
                [21, 21, 22, 22]
            ])
        # For task_id 2, use a larger task board (8x6)
        elif task_id == 2:
            task_board = np.array([
                [1, 2, 3, 4, 5, 6, 7, 8],
                [9, 10, 11, 12, 9, 10, 11, 12],
                [17, 18, 19, 20, 17, 18, 19, 20],
                [5, 6, 7, 8, 1, 2, 3, 4],
                [13, 14, 15, 16, 13, 14, 15, 16],
                [21, 21, 22, 22, 21, 21, 22, 22]
            ])
        else:
            # Default to the first task board if task_id is not found
            task_board = np.array([
                [1, 2, 3, 4],
                [9, 10, 11, 12],
                [17, 18, 19, 20],
                [5, 6, 7, 8],
                [13, 14, 15, 16],
                [21, 21, 22, 22]
            ])
        
        # Task properties define the characteristics of each task
        # type 1: Leader-specific tasks (unbolting by Franka)
        # type 2: Follower1-specific tasks (pick-and-place by UR10)
        # type 3: Follower2-specific tasks (casing work by Kuka)
        # type 4: Collaborative tasks between Follower1 and Follower2
        # type 5: Tasks requiring Leader and one Follower
        # type 6: Complex tasks requiring all three robots
        
        # Create a type array matching the size of the largest task ID
        max_task_id = np.max(task_board)
        type_array = np.zeros(max_task_id + 1, dtype=int)
        
        # Assign task types
        type_array[1:9] = 1      # Franka tasks (screws)
        type_array[9:13] = 2     # UR10 tasks (battery cells)
        type_array[13:17] = 3    # Kuka tasks (casing)
        type_array[17:21] = 4    # UR10 + Kuka collaborative tasks (connectors)
        type_array[21:23] = 6    # All three robots (complex assemblies)
        
        # Success probabilities for each robot on different task types
        l_succ = np.zeros(max_task_id + 1)
        f1_succ = np.zeros(max_task_id + 1)
        f2_succ = np.zeros(max_task_id + 1)
        
        # Set success probabilities based on task types
        # Type 1: Leader (Franka) tasks
        l_succ[type_array == 1] = 0.9
        f1_succ[type_array == 1] = 0.0
        f2_succ[type_array == 1] = 0.0
        
        # Type 2: Follower 1 (UR10) tasks
        l_succ[type_array == 2] = 0.0
        f1_succ[type_array == 2] = 0.9
        f2_succ[type_array == 2] = 0.0
        
        # Type 3: Follower 2 (Kuka) tasks
        l_succ[type_array == 3] = 0.0
        f1_succ[type_array == 3] = 0.0
        f2_succ[type_array == 3] = 0.9
        
        # Type 4: Follower 1 + Follower 2 collaborative tasks
        l_succ[type_array == 4] = 0.0
        f1_succ[type_array == 4] = 0.7
        f2_succ[type_array == 4] = 0.7
        
        # Type 5: Leader + Follower collaborative tasks (not in this board)
        
        # Type 6: All three robots collaborative tasks
        l_succ[type_array == 6] = 0.7
        f1_succ[type_array == 6] = 0.7
        f2_succ[type_array == 6] = 0.7
        
        # Shape indicates the physical size/complexity (affects timing)
        shape_array = np.ones(max_task_id + 1, dtype=int)
        shape_array[0] = 0  # Empty space has no shape
        shape_array[type_array == 6] = 3  # Complex tasks have larger shape value
        
        task_prop = {
            'type': type_array,
            'shape': shape_array,
            'l_succ': l_succ,
            'f1_succ': f1_succ,
            'f2_succ': f2_succ
        }
        
        return task_board, task_prop

    def get_task_info(self) -> Dict:
        """
        Get task information for initializing the learning algorithms.
        
        Returns:
            Dictionary containing task information
        """
        info = {}
        info['task_id'] = self.task_id
        info['dims'] = self.task_board.shape[1]
        info['dimAl'] = self.task_board.shape[1] + 1   # +1 for "do nothing" action
        info['dimAf1'] = self.task_board.shape[1] + 1  # +1 for "do nothing" action
        info['dimAf2'] = self.task_board.shape[1] + 1  # +1 for "do nothing" action
        info['dimal'] = 1
        info['dimaf1'] = 1
        info['dimaf2'] = 1
        info['task_prop'] = self.task_prop
        info['board_shape'] = self.task_board.shape
        return info

    def get_current_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the current state of the environment.
        
        Returns:
            Tuple containing:
            - First row of the current board (simplified state representation)
            - Complete current board
        """
        return np.copy(self.curr_board[0, :]), np.copy(self.curr_board)
    
    def get_action_masks(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get action masks for all three robots.
        
        Returns:
            Tuple containing boolean masks for valid actions for each robot:
            - leader_mask: Mask for leader robot (True if action is valid)
            - follower1_mask: Mask for follower1 robot (True if action is valid)
            - follower2_mask: Mask for follower2 robot (True if action is valid)
        """
        # Get the current first row (available tasks)
        first_row = self.curr_board[0, :]
        
        # Initialize masks with True for "do nothing" action (-1)
        leader_mask = np.zeros(self.task_board.shape[1] + 1, dtype=bool)
        follower1_mask = np.zeros(self.task_board.shape[1] + 1, dtype=bool)
        follower2_mask = np.zeros(self.task_board.shape[1] + 1, dtype=bool)
        
        # "Do nothing" action is always valid (action index 0 corresponds to -1)
        leader_mask[0] = True
        follower1_mask[0] = True
        follower2_mask[0] = True
        
        # Check each task in the first row
        for col_idx, task_id in enumerate(first_row):
            # Skip empty spaces (task_id = 0)
            if task_id == 0:
                continue
            
            # Check if the task is feasible for each robot
            if self.is_task_feasible(task_id, 'leader'):
                leader_mask[col_idx + 1] = True  # +1 because index 0 is "do nothing"
                
            if self.is_task_feasible(task_id, 'follower1'):
                follower1_mask[col_idx + 1] = True  # +1 because index 0 is "do nothing"
                
            if self.is_task_feasible(task_id, 'follower2'):
                follower2_mask[col_idx + 1] = True  # +1 because index 0 is "do nothing"
        
        return leader_mask, follower1_mask, follower2_mask
    
    def get_enhanced_state(self) -> np.ndarray:
        """
        Get an enhanced state representation including:
        - Flattened board
        - Robot positions
        - Time budget
        - Action masks
        
        Returns:
            Enhanced state representation as a numpy array
        """
        # Flatten the current board
        flattened_board = self.curr_board.flatten()
        
        # Get robot positions (normalized between 0 and 1)
        franka_pos_norm = (self.franka_state['position'] + 1) / 2  # Assuming positions range from -1 to 1
        ur10_pos_norm = (self.ur10_state['position'] + 1) / 2
        kuka_pos_norm = (self.kuka_state['position'] + 1) / 2
        
        # Time budget (normalized between 0 and 1)
        time_budget = (self.max_time_steps - self.time_step) / self.max_time_steps
        
        # Get action masks
        leader_mask, follower1_mask, follower2_mask = self.get_action_masks()
        
        # Combine all the information
        enhanced_state = np.concatenate([
            flattened_board,
            franka_pos_norm,
            ur10_pos_norm,
            kuka_pos_norm,
            np.array([time_budget]),
            leader_mask.astype(float),
            follower1_mask.astype(float),
            follower2_mask.astype(float)
        ])
        
        return enhanced_state
    
    def set_env(self, board: np.ndarray) -> None:
        """
        Set the environment to a specific board configuration.
        
        Args:
            board: New board configuration
        """
        self.curr_board = np.copy(board)
    
    def reset_env(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Reset the environment to the initial state.
        
        Returns:
            Tuple containing:
            - First row of the current board
            - Complete current board
            - Dictionary with action masks for each robot
        """
        self.curr_board = np.copy(self.task_board)
        self.completed_tasks = []
        self.time_step = 0
        self.franka_state = {'position': self.franka_pos, 'gripper_open': True, 'holding': None}
        self.ur10_state = {'position': self.ur10_pos, 'suction_active': False, 'holding': None}
        self.kuka_state = {'position': self.kuka_pos, 'tool_active': False, 'holding': None}
        
        first_row, full_board = self.get_current_state()
        leader_mask, follower1_mask, follower2_mask = self.get_action_masks()
        
        info = {
            'action_masks': {
                'leader': leader_mask,
                'follower1': follower1_mask,
                'follower2': follower2_mask
            },
            'enhanced_state': self.get_enhanced_state()
        }
        
        return first_row, full_board, info
    
    def step(self, al: int, af1: int, af2: int) -> Tuple[np.ndarray, np.ndarray, float, float, float, bool, Dict]:
        """
        Execute one step in the environment based on all three robots' actions.
        
        Args:
            al: Leader action (Franka robot), -1 for "do nothing"
            af1: Follower 1 action (UR10 robot), -1 for "do nothing"
            af2: Follower 2 action (Kuka robot), -1 for "do nothing"
        
        Returns:
            Tuple containing:
            - First row of the updated board
            - Complete updated board
            - Reward for the leader
            - Reward for follower 1
            - Reward for follower 2
            - Whether the episode is done
            - Dictionary with additional information
        """
        # Get the current state before acting
        state, _ = self.get_current_state()
        
        # Store initial number of completed tasks
        initial_completed_tasks = len(self.completed_tasks)
        
        # Simulate if task is completed by the leader (Franka)
        if al == -1:
            tl, tl_done = 0, False  # Leader does nothing
        else:
            tl = self.curr_board[0, al]
            if tl == 0:
                tl_done = False  # Task already completed or invalid
            else:
                # Check if task is within Franka's capabilities and workspace
                if self.is_task_feasible(tl, 'leader'):
                    # Sample success based on success probability
                    tl_done = self.rng.random() < self.task_prop['l_succ'][tl]
                else:
                    tl_done = False
        
        # Simulate if task is completed by follower 1 (UR10)
        if af1 == -1:
            tf1, tf1_done = 0, False  # Follower 1 does nothing
        else:
            tf1 = self.curr_board[0, af1]
            if tf1 == 0:
                tf1_done = False  # Task already completed or invalid
            else:
                # Check if task is within UR10's capabilities and workspace
                if self.is_task_feasible(tf1, 'follower1'):
                    # Sample success based on success probability
                    tf1_done = self.rng.random() < self.task_prop['f1_succ'][tf1]
                else:
                    tf1_done = False
        
        # Simulate if task is completed by follower 2 (Kuka)
        if af2 == -1:
            tf2, tf2_done = 0, False  # Follower 2 does nothing
        else:
            tf2 = self.curr_board[0, af2]
            if tf2 == 0:
                tf2_done = False  # Task already completed or invalid
            else:
                # Check if task is within Kuka's capabilities and workspace
                if self.is_task_feasible(tf2, 'follower2'):
                    # Sample success based on success probability
                    tf2_done = self.rng.random() < self.task_prop['f2_succ'][tf2]
                else:
                    tf2_done = False
        
        # Update the task board based on the simulated results
        self.update_board(tl, tl_done, tf1, tf1_done, tf2, tf2_done)
        
        # Update robot positions based on actions
        if tl_done or al != -1:
            self.update_robot_position('leader', al)
        
        if tf1_done or af1 != -1:
            self.update_robot_position('follower1', af1)
            
        if tf2_done or af2 != -1:
            self.update_robot_position('follower2', af2)
        
        # Increment time step
        self.time_step += 1
        
        # Get rewards
        rl, rf1, rf2 = self.reward(state, al, af1, af2)
        
        # Add shaped rewards based on tasks completed in this step
        tasks_completed = len(self.completed_tasks) - initial_completed_tasks
        rl += tasks_completed * self.reward_task_cleared
        rf1 += tasks_completed * self.reward_task_cleared
        rf2 += tasks_completed * self.reward_task_cleared
        
        # Add step penalty
        rl -= self.reward_step_penalty
        rf1 -= self.reward_step_penalty
        rf2 -= self.reward_step_penalty
        
        # Check if the episode is done
        done = self.is_done() or self.time_step >= self.max_time_steps
        
        # Get updated state
        next_first_row, next_full_board = self.get_current_state()
        
        # Get action masks for the next state
        leader_mask, follower1_mask, follower2_mask = self.get_action_masks()
        
        # Return the results
        info = {
            'action_masks': {
                'leader': leader_mask,
                'follower1': follower1_mask,
                'follower2': follower2_mask
            },
            'tasks_completed': tasks_completed,
            'total_completed': len(self.completed_tasks),
            'time_step': self.time_step,
            'enhanced_state': self.get_enhanced_state()
        }
        
        return next_first_row, next_full_board, rl, rf1, rf2, done, info
    
    def is_task_feasible(self, task_id: int, robot: str) -> bool:
        """
        Check if a task is feasible for the given robot based on capabilities and workspace constraints.
        
        Args:
            task_id: ID of the task to check
            robot: 'leader', 'follower1', or 'follower2'
        
        Returns:
            Boolean indicating if the task is feasible
        """
        # Check robot capability based on task type
        task_type = self.task_prop['type'][task_id]
        
        if robot == 'leader':
            # Leader can do type 1 tasks, and participate in type 5 & 6 collaborative tasks
            if task_type not in [1, 5, 6]:
                return False
                
            # Check if Franka can reach the task
            task_pos = self.get_task_position(task_id)
            if task_pos is None:
                return False
            dist = np.linalg.norm(task_pos - self.franka_state['position'])
            return dist <= self.franka_workspace_radius
            
        elif robot == 'follower1':
            # Follower1 can do type 2 tasks, and participate in type 4, 5 (with leader), & 6 collaborative tasks
            if task_type not in [2, 4, 5, 6]:
                return False
                
            # Check if UR10 can reach the task
            task_pos = self.get_task_position(task_id)
            if task_pos is None:
                return False
            dist = np.linalg.norm(task_pos - self.ur10_state['position'])
            return dist <= self.ur10_workspace_radius
            
        elif robot == 'follower2':
            # Follower2 can do type 3 tasks, and participate in type 4, 5 (with leader), & 6 collaborative tasks
            if task_type not in [3, 4, 5, 6]:
                return False
                
            # Check if Kuka can reach the task
            task_pos = self.get_task_position(task_id)
            if task_pos is None:
                return False
            dist = np.linalg.norm(task_pos - self.kuka_state['position'])
            return dist <= self.kuka_workspace_radius
            
        return False
    
    def get_task_position(self, task_id: int) -> Optional[np.ndarray]:
        """
        Get the 3D position of a task based on its ID.
        
        In a realistic scenario, this would map task IDs to actual 
        positions on the battery module.
        
        Args:
            task_id: ID of the task
        
        Returns:
            3D position of the task, or None if the task is not found
        """
        # Find the task coordinates in the board
        coords = np.argwhere(self.curr_board == task_id)
        if len(coords) == 0:
            return None
        
        row, col = coords[0]
        
        # Map the 2D coordinates to 3D positions relative to the battery position
        x = self.battery_pos[0] + (col - self.curr_board.shape[1]/2) * 0.1
        y = self.battery_pos[1] + (row - self.curr_board.shape[0]/2) * 0.1
        z = self.battery_pos[2] + 0.05  # Slight offset from battery surface
        
        return np.array([x, y, z])
    
    def update_robot_position(self, robot: str, action: int) -> None:
        """
        Update the position of a robot based on its action.
        
        Args:
            robot: 'leader', 'follower1', or 'follower2'
            action: The robot's action
        """
        if action == -1:
            # No movement for "do nothing" action
            return
        
        task_pos = self.get_task_position(self.curr_board[0, action])
        if task_pos is None:
            # No valid task position
            return
        
        if robot == 'leader':
            # Move Franka to the task position
            self.franka_state['position'] = task_pos
            # Update gripper state based on task type
            task_id = self.curr_board[0, action]
            if task_id > 0:
                task_type = self.task_prop['type'][task_id]
                self.franka_state['gripper_open'] = task_type != 1  # Close gripper for unbolting
                self.franka_state['holding'] = task_id if task_type == 1 else None
        
        elif robot == 'follower1':
            # Move UR10 to the task position
            self.ur10_state['position'] = task_pos
            # Update suction state based on task type
            task_id = self.curr_board[0, action]
            if task_id > 0:
                task_type = self.task_prop['type'][task_id]
                self.ur10_state['suction_active'] = task_type == 2  # Activate suction for pick-and-place
                self.ur10_state['holding'] = task_id if task_type == 2 else None
        
        elif robot == 'follower2':
            # Move Kuka to the task position
            self.kuka_state['position'] = task_pos
        
    def update_board(self, tl: int, tl_done: bool, tf1: int, tf1_done: bool, tf2: int, tf2_done: bool) -> None:
        """
        Update the task board based on completed tasks.
        
        Args:
            tl: Task ID attempted by the leader
            tl_done: Whether the leader completed the task
            tf1: Task ID attempted by follower 1
            tf1_done: Whether follower 1 completed the task
            tf2: Task ID attempted by follower 2
            tf2_done: Whether follower 2 completed the task
        """
        # Process leader's task
        if tl > 0 and tl_done:
            # Find the task on the board
            positions = np.where(self.curr_board == tl)
            if len(positions[0]) > 0:
                # Mark the task as completed
                row, col = positions[0][0], positions[1][0]
                self.curr_board[row, col] = 0
                self.completed_tasks.append(tl)
                
                # Check if a row is completed
                if np.all(self.curr_board[row, :] == 0):
                    # Shift rows above down
                    for r in range(row, 0, -1):
                        self.curr_board[r, :] = self.curr_board[r-1, :]
                    # Clear the top row
                    self.curr_board[0, :] = 0
        
        # Process follower 1's task
        if tf1 > 0 and tf1_done:
            # Find the task on the board
            positions = np.where(self.curr_board == tf1)
            if len(positions[0]) > 0:
                # Mark the task as completed
                row, col = positions[0][0], positions[1][0]
                self.curr_board[row, col] = 0
                self.completed_tasks.append(tf1)
                
                # Check if a row is completed
                if np.all(self.curr_board[row, :] == 0):
                    # Shift rows above down
                    for r in range(row, 0, -1):
                        self.curr_board[r, :] = self.curr_board[r-1, :]
                    # Clear the top row
                    self.curr_board[0, :] = 0
        
        # Process follower 2's task
        if tf2 > 0 and tf2_done:
            # Find the task on the board
            positions = np.where(self.curr_board == tf2)
            if len(positions[0]) > 0:
                # Mark the task as completed
                row, col = positions[0][0], positions[1][0]
                self.curr_board[row, col] = 0
                self.completed_tasks.append(tf2)
                
                # Check if a row is completed
                if np.all(self.curr_board[row, :] == 0):
                    # Shift rows above down
                    for r in range(row, 0, -1):
                        self.curr_board[r, :] = self.curr_board[r-1, :]
                    # Clear the top row
                    self.curr_board[0, :] = 0
        
        # Special case for collaborative tasks (type 4, 5, 6)
        # Collaborative task between followers (type 4)
        if tf1 > 0 and tf2 > 0 and tf1 == tf2 and self.task_prop['type'][tf1] == 4:
            # Both followers need to work on the same task
            if tf1_done and tf2_done:
                # Find the task on the board
                positions = np.where(self.curr_board == tf1)
                if len(positions[0]) > 0:
                    # Mark the task as completed
                    row, col = positions[0][0], positions[1][0]
                    self.curr_board[row, col] = 0
                    if tf1 not in self.completed_tasks:  # Avoid double counting
                        self.completed_tasks.append(tf1)
                    
                    # Check if a row is completed
                    if np.all(self.curr_board[row, :] == 0):
                        # Shift rows above down
                        for r in range(row, 0, -1):
                            self.curr_board[r, :] = self.curr_board[r-1, :]
                        # Clear the top row
                        self.curr_board[0, :] = 0
        
        # Collaborative task between leader and a follower (type 5)
        if tl > 0 and (tf1 == tl or tf2 == tl) and self.task_prop['type'][tl] == 5:
            follower_done = (tf1 == tl and tf1_done) or (tf2 == tl and tf2_done)
            if tl_done and follower_done:
                # Find the task on the board
                positions = np.where(self.curr_board == tl)
                if len(positions[0]) > 0:
                    # Mark the task as completed
                    row, col = positions[0][0], positions[1][0]
                    self.curr_board[row, col] = 0
                    if tl not in self.completed_tasks:  # Avoid double counting
                        self.completed_tasks.append(tl)
                    
                    # Check if a row is completed
                    if np.all(self.curr_board[row, :] == 0):
                        # Shift rows above down
                        for r in range(row, 0, -1):
                            self.curr_board[r, :] = self.curr_board[r-1, :]
                        # Clear the top row
                        self.curr_board[0, :] = 0
        
        # Complex tasks requiring all three robots (type 6)
        if tl > 0 and tf1 > 0 and tf2 > 0 and tl == tf1 and tl == tf2 and self.task_prop['type'][tl] == 6:
            if tl_done and tf1_done and tf2_done:
                # Find the task on the board
                positions = np.where(self.curr_board == tl)
                if len(positions[0]) > 0:
                    # Mark the task as completed
                    row, col = positions[0][0], positions[1][0]
                    self.curr_board[row, col] = 0
                    if tl not in self.completed_tasks:  # Avoid double counting
                        self.completed_tasks.append(tl)
                    
                    # Check if a row is completed
                    if np.all(self.curr_board[row, :] == 0):
                        # Shift rows above down
                        for r in range(row, 0, -1):
                            self.curr_board[r, :] = self.curr_board[r-1, :]
                        # Clear the top row
                        self.curr_board[0, :] = 0
    
    def reward(self, state: np.ndarray, al: int, af1: int, af2: int) -> Tuple[float, float, float]:
        """
        Calculate rewards for all three robots based on their actions.
        
        Args:
            state: State before taking actions
            al: Leader action
            af1: Follower 1 action
            af2: Follower 2 action
        
        Returns:
            Tuple of rewards for leader, follower1, and follower2
        """
        # Base reward (small penalty for each step)
        rl = -self.reward_step_penalty
        rf1 = -self.reward_step_penalty
        rf2 = -self.reward_step_penalty
        
        # Check if there was a collision in action selection (multiple robots choosing the same task)
        collision = False
        if al != -1 and af1 == al:
            collision = True
        if al != -1 and af2 == al:
            collision = True
        if af1 != -1 and af2 == af1:
            # Exception for collaborative tasks
            if al != -1 and al != af1:
                # Check if it's a collaborative task (type 4)
                task_id = self.curr_board[0, af1]
                if task_id > 0 and self.task_prop['type'][task_id] != 4:
                    collision = True
        
        # Apply collision penalty if applicable
        if collision:
            rl -= self.reward_collision_penalty
            rf1 -= self.reward_collision_penalty
            rf2 -= self.reward_collision_penalty
        
        # Reward for attempting appropriate tasks based on robot capabilities
        if al != -1:
            task_id = self.curr_board[0, al]
            if task_id > 0:
                task_type = self.task_prop['type'][task_id]
                # Leader is good at type 1, 5, and 6 tasks
                if task_type == 1:
                    rl += 0.2
                elif task_type == 5 or task_type == 6:
                    rl += 0.1
        
        if af1 != -1:
            task_id = self.curr_board[0, af1]
            if task_id > 0:
                task_type = self.task_prop['type'][task_id]
                # Follower 1 is good at type 2, 4, 5, and 6 tasks
                if task_type == 2:
                    rf1 += 0.2
                elif task_type == 4 or task_type == 5 or task_type == 6:
                    rf1 += 0.1
        
        if af2 != -1:
            task_id = self.curr_board[0, af2]
            if task_id > 0:
                task_type = self.task_prop['type'][task_id]
                # Follower 2 is good at type 3, 4, 5, and 6 tasks
                if task_type == 3:
                    rf2 += 0.2
                elif task_type == 4 or task_type == 5 or task_type == 6:
                    rf2 += 0.1
        
        # Check if a row was completed (by comparing previous state with current)
        rows_before = np.count_nonzero(np.any(state > 0, axis=1))
        rows_after = np.count_nonzero(np.any(self.curr_board > 0, axis=1))
        
        if rows_after < rows_before:
            # Row completion bonus
            row_bonus = self.reward_row_bonus
            rl += row_bonus
            rf1 += row_bonus
            rf2 += row_bonus
        
        return rl, rf1, rf2

    def is_done(self) -> bool:
        """
        Check if the episode is done (all tasks completed).
        
        Returns:
            Boolean indicating if the episode is done
        """
        # Episode is done when all cells in the task board are empty (0)
        return np.all(self.curr_board == 0)

    def render(self, ax=None):
        """
        Render the environment.
        
        Args:
            ax: Matplotlib axis for rendering, if None, a new one will be created
        """
        if ax is None:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
        
        # Clear the axis
        ax.clear()
        
        # Set up the plot
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Battery Disassembly Environment')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([0, 1])
        
        # Plot the battery module
        ax.scatter(self.battery_pos[0], self.battery_pos[1], self.battery_pos[2], 
                color='gray', s=200, label='Battery Module')
        
        # Plot the task board
        board_height, board_width = self.curr_board.shape
        for row in range(board_height):
            for col in range(board_width):
                task_id = self.curr_board[row, col]
                if task_id > 0:
                    # Get task position
                    task_pos = self.get_task_position(task_id)
                    if task_pos is not None:
                        # Color based on task type
                        task_type = self.task_prop['type'][task_id]
                        color = 'blue'  # Default
                        if task_type == 1:
                            color = 'red'  # Leader tasks
                        elif task_type == 2:
                            color = 'green'  # Follower 1 tasks
                        elif task_type == 3:
                            color = 'purple'  # Follower 2 tasks
                        elif task_type == 4:
                            color = 'orange'  # Collaborative tasks (followers)
                        elif task_type == 5:
                            color = 'brown'  # Collaborative tasks (leader + follower)
                        elif task_type == 6:
                            color = 'black'  # Complex tasks (all three robots)
                        
                        # Plot the task
                        ax.scatter(task_pos[0], task_pos[1], task_pos[2], 
                                color=color, s=100, alpha=0.7)
                        
                        # Add task ID label
                        ax.text(task_pos[0], task_pos[1], task_pos[2], 
                            str(task_id), color='white', fontsize=8, 
                            horizontalalignment='center', verticalalignment='center')
        
        # Plot the robots
        ax.scatter(self.franka_state['position'][0], self.franka_state['position'][1], self.franka_state['position'][2], 
                color='red', s=150, label='Franka (Leader)')
        ax.scatter(self.ur10_state['position'][0], self.ur10_state['position'][1], self.ur10_state['position'][2], 
                color='green', s=150, label='UR10 (Follower 1)')
        ax.scatter(self.kuka_state['position'][0], self.kuka_state['position'][1], self.kuka_state['position'][2], 
                color='purple', s=150, label='Kuka (Follower 2)')
        
        # Plot the bins
        for bin_name, bin_pos in self.bin_positions.items():
            ax.scatter(bin_pos[0], bin_pos[1], bin_pos[2], 
                    color='cyan', s=100, alpha=0.5)
            ax.text(bin_pos[0], bin_pos[1], bin_pos[2] + 0.05, 
                bin_name, color='black', fontsize=8, 
                horizontalalignment='center')
        
        # Add legend
        ax.legend()
        
        # Add episode info
        info_text = f"Time Step: {self.time_step}\n"
        info_text += f"Tasks Completed: {len(self.completed_tasks)}\n"
        info_text += f"Remaining Tasks: {np.count_nonzero(self.curr_board > 0)}\n"
        
        # Add robot states
        info_text += f"\nFranka: {'Gripper Closed' if not self.franka_state['gripper_open'] else 'Gripper Open'}\n"
        info_text += f"UR10: {'Suction Active' if self.ur10_state['suction_active'] else 'Suction Inactive'}\n"
        info_text += f"Kuka: {'Tool Active' if self.kuka_state['tool_active'] else 'Tool Inactive'}\n"
        
        # Add text annotation
        ax.text2D(0.02, 0.95, info_text, transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
        
        # Draw and update
        plt.draw()
        
        return ax