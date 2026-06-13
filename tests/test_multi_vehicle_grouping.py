"""MultiVehiclePhysics 의 그룹핑/입력분배 로직 순수-Python 단위 테스트.

전체 물리(GPU)는 samples/perf_l2_l3_combined.py 등 통합 테스트가 커버한다.
여기서는 GPU 없이 검증 가능한 부분 — cfg 정체성 기준 kind 묶기, 호출 순서
보존, flat→(kind,slot) 매핑, 입력 재분배 — 만 검증한다. 이 인덱스 재배치가
L2×L3 의 K×N 입력 라우팅이 의존하는 off-by-one 위험 지점이다.
"""
from genesis_vehicle.multi_vehicle import (
    group_vehicles_by_cfg, rebucket_inputs,
)


def _veh(cfg):
    # (entity, sensor, cfg) — 그룹핑은 cfg 정체성만 보므로 앞 둘은 더미
    return (object(), object(), cfg)


def test_single_kind_groups_all_into_one():
    cfg = object()
    vehicles = [_veh(cfg), _veh(cfg), _veh(cfg)]
    group_order, groups, flat_to_kind = group_vehicles_by_cfg(vehicles)
    assert len(group_order) == 1
    assert len(groups[group_order[0]]) == 3
    # 단일 kind: flat i → (i 번째, kind 0, slot i)
    assert flat_to_kind == [(0, 0, 0), (1, 0, 1), (2, 0, 2)]


def test_interleaved_kinds_preserve_caller_order_within_kind():
    A, B = object(), object()
    # 호출 순서: A B A B A  → kind 0 = A(슬롯 0,1,2), kind 1 = B(슬롯 0,1)
    vehicles = [_veh(A), _veh(B), _veh(A), _veh(B), _veh(A)]
    group_order, groups, flat_to_kind = group_vehicles_by_cfg(vehicles)
    assert group_order == [id(A), id(B)]          # first-seen 순서 = kind 순서
    assert len(groups[id(A)]) == 3
    assert len(groups[id(B)]) == 2
    # flat_to_kind 는 flat_i 오름차순, 각 항목이 올바른 (kind, slot) 인지
    expected = [
        (0, 0, 0),   # flat0 = A → kindA slot0
        (1, 1, 0),   # flat1 = B → kindB slot0
        (2, 0, 1),   # flat2 = A → kindA slot1
        (3, 1, 1),   # flat3 = B → kindB slot1
        (4, 0, 2),   # flat4 = A → kindA slot2
    ]
    assert flat_to_kind == expected


def test_rebucket_routes_each_flat_input_to_its_slot():
    A, B = object(), object()
    vehicles = [_veh(A), _veh(B), _veh(A), _veh(B), _veh(A)]
    _order, _groups, flat_to_kind = group_vehicles_by_cfg(vehicles)
    kind_sizes = [3, 2]   # kindA=3, kindB=2
    inputs = ["a0", "b0", "a1", "b1", "a2"]   # caller flat order
    per_kind = rebucket_inputs(inputs, flat_to_kind, kind_sizes)
    assert per_kind == [["a0", "a1", "a2"], ["b0", "b1"]]


def test_rebucket_round_trips_through_grouping():
    # 임의 구성에서 모든 입력이 정확히 한 번씩, 올바른 슬롯에 도착하는지
    cfgs = [object(), object(), object()]
    pattern = [0, 0, 1, 2, 1, 0, 2]    # cfg 인덱스 시퀀스
    vehicles = [_veh(cfgs[i]) for i in pattern]
    group_order, groups, flat_to_kind = group_vehicles_by_cfg(vehicles)
    kind_sizes = [len(groups[k]) for k in group_order]
    inputs = [f"v{i}" for i in range(len(vehicles))]
    per_kind = rebucket_inputs(inputs, flat_to_kind, kind_sizes)
    # 모든 입력이 평탄화 시 정확히 한 번씩 등장
    flat_back = [x for kind in per_kind for x in kind]
    assert sorted(flat_back) == sorted(inputs)
    assert all(x is not None for x in flat_back)
    # 슬롯 수 합 == 차량 수
    assert sum(kind_sizes) == len(vehicles)


def test_single_vehicle():
    cfg = object()
    group_order, groups, flat_to_kind = group_vehicles_by_cfg([_veh(cfg)])
    assert flat_to_kind == [(0, 0, 0)]
    assert rebucket_inputs(["x"], flat_to_kind, [1]) == [["x"]]
