"""
Chameleon Soft Actuator Dynamics Model.

This module provides the kinematic and dynamic models for a 2-DOF extensible 
soft actuator (chameleon tongue). It calculates forward kinematics, the equations 
of motion (Inertia, Coriolis, Gravity matrices), and forward dynamics.
"""

import math
import numpy as np

class Chameleon:
    """
    Dynamic plant model for a 2-DOF chameleon tongue actuator.
    
    The system consists of an angular degree of freedom (q[0]) and an 
    extensible length degree of freedom (q[1]).
    """
    
    def __init__(self):
        """
        Initializes the plant model with physical properties and default states.
        """
        # Dimensionality
        self.dimq = 2  # Joint space: [angle, length]
        self.dimr = 2  # Task space: [x, y]
        
        # Physical parameters
        self.m = 0.1   # Mass of the tongue [kg]
        self.g = 9.81  # Gravity acceleration [m/s^2]
        
        # State vectors
        self.q = np.zeros(2)
        self.qdot = np.zeros(2)
        self.tau = np.zeros(2)

    def set_m(self, m: float):
        """
        Updates the mass of the actuator.

        Args:
            m (float): New mass value in kilograms.
        """
        self.m = m

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """
        Computes the end-effector (tongue tip) position in task space.

        The forward kinematics are defined as:
        $$r_0 = q_1 \cos(q_0)$$
        $$r_1 = q_1 \sin(q_0)$$

        Args:
            q (np.ndarray): Joint configuration array [angle, length].

        Returns:
            np.ndarray: Task space coordinates [x, y].
        """
        r = np.empty(self.dimr)
        r[0] = q[1] * math.cos(q[0])
        r[1] = q[1] * math.sin(q[0])
        return r
        
    def get_MCG(self, q: np.ndarray, qdot: np.ndarray) -> tuple:
        """
        Calculates the dynamic equation matrices for the current state.

        Args:
            q (np.ndarray): Joint angles and extensions.
            qdot (np.ndarray): Joint velocities.

        Returns:
            tuple: A tuple containing:
                - M (np.ndarray): Inertia matrix (2x2).
                - C (np.ndarray): Coriolis and centrifugal matrix (2x2).
                - G (np.ndarray): Gravity vector (2,).
        """
        m = self.m
        g = self.g
        
        M = np.array([
            [m * q[1]**2, 0], 
            [0,           m]
        ])
        
        C = np.array([
            [0,                   2 * m * q[1] * qdot[0]], 
            [-m * q[1] * qdot[0], 0                     ]
        ])
        
        G = np.array([
            m * g * q[1] * math.cos(q[0]), 
            m * g * math.sin(q[0])
        ])
        
        return M, C, G
    
    def get_joint_acceleration(self, q: np.ndarray, qdot: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """
        Computes the forward dynamics (joint accelerations) given current state and torques.

        Solves the equation of motion:
        $$\ddot{q} = M^{-1} (\tau - C \dot{q} - G)$$

        Args:
            q (np.ndarray): Joint angles and extensions.
            qdot (np.ndarray): Joint velocities.
            tau (np.ndarray): Applied joint generalized forces/torques.

        Returns:
            np.ndarray: Computed joint accelerations.
        """
        M, C, G = self.get_MCG(q, qdot)
        
        # Uses np.linalg.solve instead of explicit matrix inversion for numerical stability
        qddot = np.linalg.solve(M, tau - C @ qdot - G).ravel()
        return qddot
