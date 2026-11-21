# SPDX-License-Identifier: Apache-2.0
# Derived from the field port (PathFollower_MPC/mpc/alpasim_nmpc.py:57-114) of
# NVlabs/alpasim nonlinear_mpc.py (Copyright (c) 2026 NVIDIA Corporation,
# Apache-2.0); the upstream revision was not recorded by the port.
# Modified by the GenesisVehicle contributors (see NOTICE).
"""MPC prediction models.

``"bicycle"`` (:func:`build_bicycle`) is the field model moved verbatim:

* state (8) ``[x_rig, y_rig, yaw, v_cg_x, v_cg_y, yaw_rate, delta, a]`` in the
  frame of the rig (rear-axle centre) at solve time; input (2)
  ``[delta_cmd, a_cmd]`` (front steering angle, + = left; acceleration);
* ``v_cg_x < kinematic_threshold_speed`` (5.0 m/s by default): kinematic
  bicycle, lateral states pulled toward it with a first-order gain of 10;
  otherwise a linear-tyre dynamic bicycle whose per-wheel cornering stiffness
  is doubled to an axle;
* ``delta`` and ``a`` follow their commands with first-order lags
  ``steering_time_constant`` / ``acceleration_time_constant``. With the
  identified steering time constant this counts the steering lag twice (the
  model's own yaw dynamics on top of a measured yaw-rate lag); kept for
  parity with the field controller.

:func:`bicycle_rhs_numpy` is an independent numpy transcription of the same
right-hand side, used by the tests to check the casadi model.

casadi / do_mpc are imported inside :func:`build_bicycle`, never at module
import.
"""
from __future__ import annotations

import math

import numpy as np

#: Floor on ``v_cg_x`` in the dynamic branch's 1/v terms (field ``EPS_SPEED``).
EPS_SPEED = 1.0e-2
#: First-order pull of the lateral states toward the kinematic model.
KIN_GAIN = 10.0

STATE_NAMES = ("x_rig_inertial", "y_rig_inertial", "yaw_angle", "v_cg_x", "v_cg_y",
               "yaw_rate", "front_steering_angle", "acceleration")
INPUT_NAMES = ("front_steering_angle_cmd", "acceleration_cmd")


def build_bicycle(params):
    """The field ``NonlinearMPC._build_model`` (``alpasim_nmpc.py:57-114``):
    a continuous-time do_mpc ``Model`` (SX) with the tracking time-varying
    parameters ``x_ref, y_ref, heading_ref, tracking_enabled``. Set up and
    returned."""
    import casadi
    from ._deps import import_do_mpc
    do_mpc = import_do_mpc()

    model = do_mpc.model.Model("continuous", "SX")
    model.set_variable(var_type="_x", var_name="x_rig_inertial", shape=(1, 1))
    model.set_variable(var_type="_x", var_name="y_rig_inertial", shape=(1, 1))
    yaw_angle = model.set_variable(var_type="_x", var_name="yaw_angle", shape=(1, 1))
    v_cg_x = model.set_variable(var_type="_x", var_name="v_cg_x", shape=(1, 1))
    v_cg_y = model.set_variable(var_type="_x", var_name="v_cg_y", shape=(1, 1))
    yaw_rate = model.set_variable(var_type="_x", var_name="yaw_rate", shape=(1, 1))
    front_steering_angle = model.set_variable(var_type="_x", var_name="front_steering_angle", shape=(1, 1))
    acceleration = model.set_variable(var_type="_x", var_name="acceleration", shape=(1, 1))
    front_steering_angle_cmd = model.set_variable(var_type="_u", var_name="front_steering_angle_cmd")
    acceleration_cmd = model.set_variable(var_type="_u", var_name="acceleration_cmd")

    kinetic_mass = params.mass * casadi.fmax(v_cg_x, EPS_SPEED)
    kinetic_inertia = params.inertia * casadi.fmax(v_cg_x, EPS_SPEED)
    lf = params.wheelbase - params.l_rig_to_cg
    lf_caf = lf * params.front_cornering_stiffness
    lr_car = params.l_rig_to_cg * params.rear_cornering_stiffness
    lf_sq_caf = lf * lf_caf
    lr_sq_car = params.l_rig_to_cg * lr_car
    a_00 = -2 * (params.front_cornering_stiffness + params.rear_cornering_stiffness) / kinetic_mass
    a_01 = -v_cg_x - 2 * (lf_caf - lr_car) / kinetic_mass
    a_10 = -2 * (lf_caf - lr_car) / kinetic_inertia
    a_11 = -2 * (lf_sq_caf + lr_sq_car) / kinetic_inertia
    b_00 = 2 * params.front_cornering_stiffness / params.mass
    b_10 = 2 * lf_caf / params.inertia
    use_kinematic_model = casadi.lt(v_cg_x, params.kinematic_threshold_speed)

    model.set_rhs("x_rig_inertial", casadi.if_else(
        use_kinematic_model,
        v_cg_x * casadi.cos(yaw_angle),
        v_cg_x * casadi.cos(yaw_angle) - (v_cg_y - params.l_rig_to_cg * yaw_rate) * casadi.sin(yaw_angle)))
    model.set_rhs("y_rig_inertial", casadi.if_else(
        use_kinematic_model,
        v_cg_x * casadi.sin(yaw_angle),
        v_cg_x * casadi.sin(yaw_angle) + (v_cg_y - params.l_rig_to_cg * yaw_rate) * casadi.cos(yaw_angle)))
    model.set_rhs("yaw_angle", yaw_rate)
    model.set_rhs("v_cg_x", acceleration)
    yaw_rate_kinematic = v_cg_x * front_steering_angle / params.wheelbase
    v_cg_y_kinematic = v_cg_x * front_steering_angle * params.l_rig_to_cg / params.wheelbase
    GAIN = KIN_GAIN
    model.set_rhs("v_cg_y", casadi.if_else(
        use_kinematic_model, GAIN * (v_cg_y_kinematic - v_cg_y),
        a_00 * v_cg_y + a_01 * yaw_rate + b_00 * front_steering_angle))
    model.set_rhs("yaw_rate", casadi.if_else(
        use_kinematic_model, GAIN * (yaw_rate_kinematic - yaw_rate),
        a_10 * v_cg_y + a_11 * yaw_rate + b_10 * front_steering_angle))
    model.set_rhs("front_steering_angle",
                  (front_steering_angle_cmd - front_steering_angle) / params.steering_time_constant)
    model.set_rhs("acceleration", (acceleration_cmd - acceleration) / params.acceleration_time_constant)
    model.set_variable(var_type="_tvp", var_name="x_ref")
    model.set_variable(var_type="_tvp", var_name="y_ref")
    model.set_variable(var_type="_tvp", var_name="heading_ref")
    model.set_variable(var_type="_tvp", var_name="tracking_enabled")
    model.setup()
    return model


def bicycle_rhs_numpy(x, u, params):
    """The bicycle right-hand side in numpy (float64): a separate transcription
    of the field coefficients (``alpasim_nmpc.py:66-81``), the test reference
    for :func:`build_bicycle`."""
    X, Y, yaw, vx, vy, r, d, a = (float(v) for v in x)
    d_cmd, a_cmd = (float(v) for v in u)
    m, iz = params.mass, params.inertia
    L, lr = params.wheelbase, params.l_rig_to_cg
    lf = L - lr
    caf, car = params.front_cornering_stiffness, params.rear_cornering_stiffness
    if vx < params.kinematic_threshold_speed:
        dX = vx * math.cos(yaw)
        dY = vx * math.sin(yaw)
        dvy = KIN_GAIN * (vx * d * lr / L - vy)
        dr = KIN_GAIN * (vx * d / L - r)
    else:
        v_rig_y = vy - lr * r
        dX = vx * math.cos(yaw) - v_rig_y * math.sin(yaw)
        dY = vx * math.sin(yaw) + v_rig_y * math.cos(yaw)
        vs = max(vx, EPS_SPEED)
        km, ki = m * vs, iz * vs
        a00 = -2 * (caf + car) / km
        a01 = -vx - 2 * (lf * caf - lr * car) / km
        a10 = -2 * (lf * caf - lr * car) / ki
        a11 = -2 * (lf * lf * caf + lr * lr * car) / ki
        b00 = 2 * caf / m
        b10 = 2 * lf * caf / iz
        dvy = a00 * vy + a01 * r + b00 * d
        dr = a10 * vy + a11 * r + b10 * d
    dd = (d_cmd - d) / params.steering_time_constant
    da = (a_cmd - a) / params.acceleration_time_constant
    return np.array([dX, dY, r, a, dvy, dr, dd, da])
