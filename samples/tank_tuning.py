"""Tank tuning overrides — the config the bundled reference sweep table
(`data/tank_sweep_signed.csv`) is measured with.

Usable both as an import (`from genesis_vehicle.samples.tank_tuning import
TankTuning`) and as a sweep-measurement config file (module-level
`apply_config` / `apply_runtime_config`):

    python -m genesis_vehicle.control.sweep_measure \\
        --urdf genesis_vehicle/samples/urdf/tank_ray.urdf \\
        --preset tank_10w_skid_belt \\
        --config genesis_vehicle/samples/tank_tuning.py \\
        --output tank_sweep.csv --gpu

The sweep table is only valid for the exact (URDF, preset, overrides)
triple — change anything here and re-measure.
"""
from __future__ import annotations


class TankTuning:
    """The override set the bundled sweep table was measured with."""
    OMEGA_CAP = 10.4            # rad/s — wheel-omega cap = 15 km/h top speed
    T_BRAKE_MAX = 200_000.0     # N*m total — brake can out-torque tire friction
    I_WHEEL = 100.0             # kg*m^2 — damps brake-friction oscillation
    MU_LAT = 0.5                # flat lateral friction — responsive skid spin
    SFL_V_THR = 5.0             # StaticFrictionLock enables right below cruise
    SFL_K_SPRING = 1_000_000.0
    SFL_K_DAMP = 200_000.0      # ~1.35x critical -> over-damped hold

    @classmethod
    def apply_config(cls, cfg):
        from genesis_vehicle import StaticFrictionLock
        for hook in cfg.stability_hooks:
            if isinstance(hook, StaticFrictionLock):
                hook.v_thr = cls.SFL_V_THR
                hook.k_spring = cls.SFL_K_SPRING
                hook.k_damp = cls.SFL_K_DAMP
        for w in cfg.wheels:
            w.i_wheel = cls.I_WHEEL
            w.mu_lat = cls.MU_LAT

    @classmethod
    def apply_resolved(cls, resolved):
        """Post-build overrides on a resolved config. Use with
        ``Vehicle.resolved`` (works in BOTH VehicleScene solver modes) or
        ``VehiclePhysics.resolved``."""
        resolved.drivetrain.omega_max_drive = cls.OMEGA_CAP
        resolved.drivetrain.t_brake_max = cls.T_BRAKE_MAX

    @classmethod
    def apply_runtime_config(cls, physics):
        cls.apply_resolved(physics.resolved)


# Module-level entry points for `sweep_measure --config <this file>`.
def apply_config(cfg):
    TankTuning.apply_config(cfg)


def apply_runtime_config(physics):
    TankTuning.apply_runtime_config(physics)
