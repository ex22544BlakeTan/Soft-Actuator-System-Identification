"""
Soft Actuator System Identification and Data-Driven Inverse Kinematics.

This module performs dynamic parameter identification (e.g., mass estimation) using 
Least Squares Regression based on the Euler-Lagrange dynamic formulations. 
Additionally, it trains a Multi-Layer Perceptron (MLP) to learn the non-linear 
Inverse Kinematics (IK) mapping from the task space to the joint space.
"""

import math
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Note: Ensure the imported Chameleon class matches the capitalized naming convention
from chameleon import Chameleon 

# ==========================================
# Inverse Kinematics (IK) via Deep Learning
# ==========================================
def estimate_model(R: np.ndarray, Q: np.ndarray) -> MLPRegressor:
    """
    Trains an MLP neural network to approximate the non-linear Inverse Kinematics.

    Args:
        R (np.ndarray): Task space coordinates (end-effector locations), shape (2, N).
        Q (np.ndarray): Joint space configurations (angle and length), shape (2, N).

    Returns:
        MLPRegressor: Trained neural network model mapping R -> Q.
    """
    R_train = R.T
    Q_train = Q.T

    model = MLPRegressor(
        hidden_layer_sizes=(200, 200, 200),
        activation='tanh',
        solver='lbfgs',
        max_iter=10000,
        random_state=1
    )
    model.fit(R_train, Q_train)
    
    return model

def estimate_rmse(Rd: np.ndarray, model: MLPRegressor) -> float:
    """
    Evaluates the Root Mean Square Error (RMSE) of the neural network IK solver.

    Args:
        Rd (np.ndarray): Desired task space coordinates.
        model (MLPRegressor): The trained IK neural network.

    Returns:
        float: The calculated RMSE in the task space.
    """
    Rd_test = Rd.T
    Q_pred = model.predict(Rd_test)
    
    cham = Chameleon()
    R_actual = []
    
    for q in Q_pred:
        r = cham.forward_kinematics(q)
        R_actual.append(r)
    
    R_actual = np.array(R_actual)
    rmse = np.sqrt(mean_squared_error(Rd_test, R_actual))
    
    return rmse

def estimate_probability(Rd: np.ndarray, model: MLPRegressor) -> float:
    """
    Calculates the operational success rate (probability of reaching the target) 
    within a specified spatial tolerance radius.

    Args:
        Rd (np.ndarray): Target coordinates in the task space.
        model (MLPRegressor): The trained IK neural network.

    Returns:
        float: Probability (0.0 to 1.0) of successful manipulation.
    """
    Rd_test = Rd.T
    Q_pred = model.predict(Rd_test)
    
    cham = Chameleon()
    success_count = 0
    tolerance_radius = 0.01  
    
    for i in range(len(Rd_test)):
        r_actual = cham.forward_kinematics(Q_pred[i])
        distance = np.linalg.norm(r_actual - Rd_test[i])
        
        if distance <= tolerance_radius:
            success_count += 1
            
    probability = success_count / len(Rd_test)
    return probability

# ==========================================
# Dynamic System Identification
# ==========================================
def estimate_mass(Q: np.ndarray, Qdot: np.ndarray, Qddot: np.ndarray, Tau: np.ndarray) -> float:
    """
    Estimates the actuator payload/mass using Newton-Euler dynamic equations 
    and Ordinary Least Squares (OLS) regression.

    Formulates the system dynamics into a linear regression problem $\tau = \Phi m$, 
    where the regressor matrix $\Phi$ components ($p_1, p_2$) are derived as:
    
    $$p_1 = q_2^2 \ddot{q}_1 + 2q_2 \dot{q}_1 \dot{q}_2 + g q_2 \cos(q_1)$$
    $$p_2 = \ddot{q}_2 - q_2 \dot{q}_1^2 + g \sin(q_1)$$

    Args:
        Q (np.ndarray): Joint positions.
        Qdot (np.ndarray): Joint velocities.
        Qddot (np.ndarray): Joint accelerations.
        Tau (np.ndarray): Applied joint generalized forces/torques.

    Returns:
        float: Estimated mass of the physical system.
    """
    # Standardize input shapes to (N, 2)
    if Q.shape[0] == 2:
        Q, Qdot = Q.T, Qdot.T
        Qddot, Tau = Qddot.T, Tau.T

    g_const = 9.81 
    
    # [Senior Engineer Refactoring]: Vectorized Numpy operations replacing the original for-loop.
    # This guarantees massive computational speedups on large telemetry datasets.
    q1, q2 = Q[:, 0], Q[:, 1]
    dq1, dq2 = Qdot[:, 0], Qdot[:, 1]
    ddq1, ddq2 = Qddot[:, 0], Qddot[:, 1]
    tau1, tau2 = Tau[:, 0], Tau[:, 1]
    
    # Compute regressor components using matrix broadcasting
    p1 = (q2**2 * ddq1) + (2 * q2 * dq1 * dq2) + (g_const * q2 * np.cos(q1))
    p2 = ddq2 - (q2 * dq1**2) + (g_const * np.sin(q1))
    
    # Flatten into regressor matrix Phi and target vector T_flat
    Phi = np.vstack((p1, p2)).T.reshape(-1, 1)
    T_flat = np.vstack((tau1, tau2)).T.reshape(-1, 1)
    
    # Solve regression without intercept since tau = Phi * m
    reg = LinearRegression(fit_intercept=False)
    reg.fit(Phi, T_flat)
    
    return float(reg.coef_[0][0])

# ==========================================
# Execution Pipeline
# ==========================================
def main():
    np.random.seed(1)

    print("[INFO] Loading telemetry data from CSV logs...")
    try:
        R     = np.loadtxt('R.csv'    , delimiter=',') 
        Q     = np.loadtxt('Q.csv'    , delimiter=',') 
        Qdot  = np.loadtxt('Qdot.csv' , delimiter=',') 
        Qddot = np.loadtxt('Qddot.csv', delimiter=',') 
        Tau   = np.loadtxt('Tau.csv'  , delimiter=',') 
        Rd    = np.loadtxt('Rd.csv'   , delimiter=',') 
    except FileNotFoundError as e:
        print(f"[ERROR] Telemetry files missing. {e}")
        return

    print("[INFO] Training Multi-Layer Perceptron for Inverse Kinematics...")
    model = estimate_model(R, Q)

    print("[INFO] Evaluating IK performance metrics...")
    rmse = estimate_rmse(Rd, model)
    probability = estimate_probability(Rd, model)
    
    print("[INFO] Executing dynamic parameter identification...")
    mass = estimate_mass(Q, Qdot, Qddot, Tau)
    
    print("-" * 40)
    print("System Identification & Control Results:")
    print(f"  Estimated Actuator Mass : {mass:.4f} kg")
    print(f"  IK Model RMSE           : {rmse:.4f} m")
    print(f"  Target Reaching Prob.   : {probability * 100:.2f}%")
    print("-" * 40)

if __name__ == "__main__":
    main()
