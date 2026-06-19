# Soft-Actuator System Identification — Bio-Inspired "Chameleon's Tongue"

Learning an inverse model and identifying the dynamics of a soft, bio-inspired **2-DoF actuator** (a chameleon's tongue, parameterised by *angle* and *length*). Combines a neural-network inverse model with least-squares dynamic-parameter identification.

> Coursework for the Intelligence & Autonomy module (7CCEMIAA), MSc Robotics, King's College London.

## Problem
The tongue is a 2-DoF soft actuator: configuration `q = (angle, length)` maps to a tip location `r` in the plane. Given recorded trajectories — configurations `Q`, tip locations `R`, velocities `Q̇`, accelerations `Q̈`, generalised forces `τ` — and a set of prey locations `Rd`, the goals are to (1) learn to *aim* the tongue, and (2) identify its mass from the dynamics.

## Tasks

**1. Inverse-model learning** — `estimate_model`
Train an **MLP regressor** (3 × 200 hidden units, tanh, L-BFGS) mapping a desired tip location `r → q` — a learned inverse kinematics for the soft tongue.

**2. Accuracy evaluation** — `estimate_rmse`
Predict configurations for the prey locations, push them through the true **forward kinematics**, and report the **RMSE** between desired and achieved tip positions.

**3. Catch-probability estimation** — `estimate_probability`
Estimate the probability of catching prey: a strike succeeds when the achieved tip falls within a 1 cm radius of the insect.

**4. Dynamic parameter identification** — `estimate_mass`
Cast the equation of motion as a **linear-in-parameters regression** `τ = Φ(q, q̇, q̈)·m`, build the regressor `Φ` from the recorded trajectories, and solve by **least squares** to identify the tongue **mass**.

## Tech stack
Python · NumPy · scikit-learn (MLPRegressor, LinearRegression)

## Files
```
k25044931_IA_cw.py    # solution: inverse model, evaluation, mass identification
chameleon.py          # provided course scaffolding (tongue model + forward kinematics)
R.csv  Q.csv  Qdot.csv  Qddot.csv  Tau.csv  Rd.csv   # recorded trajectories & prey locations
```

## Notes
- `chameleon.py` is provided course scaffolding (the tongue forward-kinematics model).
- Coursework for 7CCEMIAA Intelligence & Autonomy, MSc Robotics, KCL.
