"""genesis_vehicle.server 서브패키지 임포트 스모크 테스트.

서버는 SDK 본체와 달리 genesis/pythonosc/psutil 런타임이 필요하므로,
해당 의존성이 없는 순수-Python CI 에서는 자동 skip 된다.
"""
import pytest


def test_server_init_is_dependency_free():
    # __init__ 은 의도적으로 빈 모듈 — pythonosc 없는 환경에서도 임포트돼야 한다.
    import genesis_vehicle.server  # noqa: F401


def test_server_modules_import_and_expose_entry_points():
    pytest.importorskip("genesis")
    pytest.importorskip("pythonosc")
    from genesis_vehicle.server import physics_server, l3_runtime, vehicle_builder

    # CLI 진입점과 두 실행 모드
    assert callable(physics_server.main)
    assert callable(physics_server.cli)
    assert callable(l3_runtime.run_l3)

    # per-entity / L3 경로가 공유하는 cfg 빌더 표면
    assert callable(vehicle_builder.build_cfg)
    assert callable(vehicle_builder.build_vehicle)
    assert callable(vehicle_builder.strip_wheel_collisions)


def test_steer_mapping_recognizes_ue_field_names():
    pytest.importorskip("genesis")
    from genesis_vehicle.server.vehicle_builder import _mapping_steer_rad

    # UE FGenesisVehicleMapping.SteerScale 은 JSON 직렬화 시 'steerScale'
    assert _mapping_steer_rad({"steerScale": 0.6}) == 0.6
    assert _mapping_steer_rad({"SteerScale": 0.5}) == 0.5
    assert _mapping_steer_rad({"maxSteerRad": 0.7}) == 0.7
    assert _mapping_steer_rad({"MaxSteerRad": 0.65}) == 0.65
    # 우선순위: maxSteerRad > steerScale
    assert _mapping_steer_rad({"maxSteerRad": 0.7, "steerScale": 0.6}) == 0.7
    assert _mapping_steer_rad({}, default=0.7) == 0.7
    assert _mapping_steer_rad({}) is None
