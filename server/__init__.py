"""genesis_vehicle.server — OSC 기반 범용 차량 물리 서버.

원본은 genesis_unreal_plugin/ 의 UE 연동 서버였으나, 프로토콜(OSC/UDP)을 말하는
어떤 클라이언트(UE, Unity, 자체 시각화 등)와도 동작하도록 SDK 서브패키지로 편입.

실행:
    python -m genesis_vehicle.server                # per-entity 모드 (이종/소수 차량, CPU)
    python -m genesis_vehicle.server --multi-env    # L3 배칭 모드 (동일 URDF 다수 차량, GPU)

추가 의존성: ``pythonosc`` (서버 전용 — SDK 본체는 필요 없음), ``psutil``,
장애물 메쉬 전처리에 ``trimesh``.

이 ``__init__`` 은 의도적으로 아무것도 임포트하지 않는다 — SDK 본체
(``import genesis_vehicle``)가 pythonosc 없는 환경에서도 동작해야 하기 때문.
서버 모듈은 ``from genesis_vehicle.server import physics_server`` 처럼 명시적으로
임포트할 것.
"""
