"""Tank tuning overrides — the config the bundled reference sweep table
(`data/tank_sweep_signed.csv`) is measured with.

Usable both as an import (`from genesis_vehicle.samples.tank_tuning import
TankTuning`) and as a sweep-measurement config file (module-level
`apply_config` / `apply_runtime_config`):

    python -m genesis_vehicle.control.sweep_measure \\
        --urdf genesis_vehicle/samples/urdf/tank_ray.urdf \\
        --preset tank_skid_belt \\
        --config genesis_vehicle/samples/tank_tuning.py \\
        --output tank_sweep.csv --gpu

The sweep table is only valid for the exact (URDF, preset, overrides)
triple AND the SDK version it was measured on — change anything here, or
upgrade the SDK, and re-measure. The bundled table was re-measured on SDK
1.8.0 (every pre-1.8.0 table is stale: 1.8.0 changed the brake, the drive
torque order and the friction ellipse).
"""
from __future__ import annotations


class TankTuning:
    """The override set the bundled sweep table was measured with."""
    OMEGA_CAP = 10.4            # rad/s — wheel-omega cap = 15 km/h top speed
    T_BRAKE_MAX = 200_000.0     # N*m total — brake can out-torque tire friction
    I_WHEEL = 100.0             # kg*m^2 — damps brake-friction oscillation
    MU_LAT = 0.5                # flat lateral friction — responsive skid spin
    # StaticFrictionLock v_thr. Since SDK 2.0.4 the engage window is
    # min(v_thr, 0.5) m/s (StaticFrictionLock.V_ENGAGE_MAX), so this 5.0 ACTS
    # AS 0.5: the lock engages on a wheel the brake holds (ctx.brake_rem > 0)
    # below 0.5 m/s planar contact speed, at any pedal, never forces the spin,
    # and takes over the tyre force only through its creep test. Through 2.0.3
    # it engaged at brake > 0.3 below 5 m/s, forced the braked wheels' spin to
    # 0 and also took over below REST_SPEED (0.05 m/s); brake_thr and
    # REST_SPEED were removed in 2.0.4. Through 1.7.0 it replaced the tyre
    # force from 5 m/s down, anchored where it engaged. Value kept.
    SFL_V_THR = 5.0
    SFL_K_SPRING = 1_000_000.0
    # Since 1.8.0 a FLOOR: the lock damps with max(k_damp, 2*sqrt(k_spring*N/g)),
    # bounded by the one-step deadbeat s*(N/g)*|v|/dt. Measured static share of
    # this tank under this config: 5780..5861 kg per wheel (10 wheels), so the
    # critical floor is 152.0k..153.1k and 200k is 1.31x critical (it is the
    # value used); the deadbeat bound caps the coefficient at about 216k..219k
    # (s 0.935, dt 0.025), above 200k. CPU/WSL2, genesis-world 1.4.5, brake 1.0
    # settle 3 s. (Through 1.7.0 this comment said "~1.35x critical" with no
    # basis given; the measured ratio is 1.31x.)
    SFL_K_DAMP = 200_000.0

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
