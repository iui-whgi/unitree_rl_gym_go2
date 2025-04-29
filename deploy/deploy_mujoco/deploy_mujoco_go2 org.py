import os
import yaml
import time
import argparse
import numpy as np
from mujoco import viewer
import mujoco

def resolve_path(path):
    # Get the environment variable
    if "{LEGGED_GYM_ROOT_DIR}" in path:
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = path.replace("{LEGGED_GYM_ROOT_DIR}", root_dir)
    return path

def quaternion_from_euler(roll, pitch, yaw):
    """
    Convert Euler angles to quaternion
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.array([w, x, y, z])

def main(args):
    if not args.endswith('.yaml'):
        args += '.yaml'

    # Load config
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", args)
    with open(cfg_path, 'r') as file:
        cfg = yaml.safe_load(file)

    # Resolve paths
    xml_path = resolve_path(cfg['xml_path'])

    # 기본 정보 출력
    print("===== CONFIGURATION =====")
    print(f"Config path: {cfg_path}")
    print(f"XML path: {xml_path}")
    print("========================\n")

    # Load model
    model = mujoco.MjModel.from_xml_path(xml_path)
    
    # 물리 파라미터 조정 (보수적인 설정)
    model.opt.timestep = 0.002
    model.opt.iterations = 8
    model.opt.solver = 2  # PGS 솔버
    model.opt.gravity[2] = -9.81  # 정확한 중력
    
    # 관절 약간 더 뻣뻣하게
    for i in range(model.njnt):
        model.jnt_stiffness[i] = 1.0  # 조인트 강성 증가
        model.dof_damping[i] = 0.5    # 적당한 댐핑
        
    # 시뮬레이션 데이터 초기화
    data = mujoco.MjData(model)

    # 모델 세부 정보 출력
    print("===== MODEL DETAILS =====")
    print(f"Number of bodies: {model.nbody}")
    print(f"Number of joints: {model.njnt}")
    print(f"Number of DOFs: {model.nq}")
    print(f"Number of actuators: {model.nu}")
    
    # 관절 및 액추에이터 이름 확인
    joint_names = []
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name:
            joint_names.append(name)
    
    actuator_names = []
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name:
            actuator_names.append(name)
    
    print(f"Joint names: {joint_names}")
    print(f"Actuator names: {actuator_names}")
    print("========================\n")

    # 매핑 생성 (수동으로 정확하게 매핑 정의)
    actuator_to_joint_map = {
        0: {'joint_name': 'FR_hip_joint', 'qpos_idx': 10, 'qvel_idx': model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'FR_hip_joint')]},
        1: {'joint_name': 'FR_thigh_joint', 'qpos_idx': 11, 'qvel_idx': model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'FR_thigh_joint')]},
        2: {'joint_name': 'FR_calf_joint', 'qpos_idx': 12, 'qvel_idx': model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'FR_calf_joint')]},
        3: {'joint_name': 'FL_hip_joint', 'qpos_idx': 7, 'qvel_idx': model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'FL_hip_joint')]},
        4: {'joint_name': 'FL_thigh_joint', 'qpos_idx': 8, 'qvel_idx': model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'FL_thigh_joint')]},
        5: {'joint_name': 'FL_calf_joint', 'qpos_idx': 9, 'qvel_idx': model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'FL_calf_joint')]},
        6: {'joint_name': 'RR_hip_joint', 'qpos_idx': 16, 'qvel_idx': model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'RR_hip_joint')]},
        7: {'joint_name': 'RR_thigh_joint', 'qpos_idx': 17, 'qvel_idx': model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'RR_thigh_joint')]},
        8: {'joint_name': 'RR_calf_joint', 'qpos_idx': 18, 'qvel_idx': model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'RR_calf_joint')]},
        9: {'joint_name': 'RL_hip_joint', 'qpos_idx': 13, 'qvel_idx': model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'RL_hip_joint')]},
        10: {'joint_name': 'RL_thigh_joint', 'qpos_idx': 14, 'qvel_idx': model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'RL_thigh_joint')]},
        11: {'joint_name': 'RL_calf_joint', 'qpos_idx': 15, 'qvel_idx': model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'RL_calf_joint')]}
    }
    
    for act_idx, info in actuator_to_joint_map.items():
        print(f"Mapped actuator {act_idx} ({actuator_names[act_idx]}) -> joint {info['joint_name']} (qpos[{info['qpos_idx']}])")
    
    # 기본 설정 불러오기
    # 중요: 뒷다리의 각도 부호를 수동으로 반전하여 모델에 맞춤
    original_default_angles = np.array(cfg['default_angles'])
    default_angles = original_default_angles.copy()
    
    # 로봇이 4다리로 서있는 안정적인 자세로 수정 (공식 문서 기준)
    # 모든 다리에 동일한 스타일 적용, 뒷다리는 앞다리와 대칭
    default_angles = np.array([
        0.0, 0.8, -1.6,  # FR: hip, thigh, calf
        0.0, 0.8, -1.6,  # FL: hip, thigh, calf
        0.0, 0.8, -1.6,  # RR: hip, thigh, calf
        0.0, 0.8, -1.6   # RL: hip, thigh, calf
    ])
    
    print(f"Original default angles: {original_default_angles}")
    print(f"Modified default angles: {default_angles}")
    
    # PD 제어기 게인
    kps = np.array(cfg.get('kps', [100.0] * model.nu))
    kds = np.array(cfg.get('kds', [5.0] * model.nu))
    print(f"kp values: {kps}")
    print(f"kd values: {kds}")
    
    # 초기 높이 설정
    initial_height = cfg.get('initial_height', 0.40)  # 약간 더 높이
    print(f"Initial height: {initial_height}m\n")
    
    # 안정화 단계 설정
    stabilization_steps = cfg.get('stabilization_steps', 1000)
    print(f"Stabilization steps: {stabilization_steps}\n")

    # 로봇 상태 초기화
    print("Initializing robot state...")
    mujoco.mj_resetData(model, data)
    
    # 초기 상태 확인
    print(f"Initial position (before setting): {data.qpos[:3]}")
    
    # 위치 및 방향 설정
    data.qpos[0:3] = [0, 0, initial_height]  # 높이 설정
    
    # 약간 기울어진 자세로 시작 (약간 뒤로 기울임)
    pitch_angle = np.radians(-5.0)  # 5도 뒤로 기울임
    quat = quaternion_from_euler(0, pitch_angle, 0)
    data.qpos[3:7] = quat
    
    # 관절 위치 설정
    for act_idx, joint_info in actuator_to_joint_map.items():
        if act_idx < len(default_angles):
            qpos_idx = joint_info['qpos_idx']
            angle = default_angles[act_idx]
            data.qpos[qpos_idx] = angle
            print(f"Set joint '{joint_info['joint_name']}' (qpos[{qpos_idx}]) to {angle}")
    
    # 속도 초기화
    data.qvel[:] = 0.0
    print("Zeroed all velocities")
    
    # 뷰어 생성
    print("\nLaunching viewer...")
    viewer_obj = viewer.launch_passive(model, data)
    viewer_obj.sync()
    
    # 초기 위치 고정 안정화
    print("\nPerforming initial stabilization with frozen base...")
    
    # 초기 위치와 방향 저장
    initial_pos = data.qpos[:3].copy()
    initial_quat = data.qpos[3:7].copy()
    
    for i in range(stabilization_steps):
        # 위치와 방향 고정
        data.qpos[:3] = initial_pos
        data.qpos[3:7] = initial_quat
        data.qvel[:6] = 0.0  # 기저 속도 제로화
        
        # 강한 PD 제어 적용
        for act_idx, joint_info in actuator_to_joint_map.items():
            if act_idx < len(default_angles):
                qpos_idx = joint_info['qpos_idx']
                qvel_idx = joint_info['qvel_idx']
                
                # 강한 게인으로 안정화
                error = default_angles[act_idx] - data.qpos[qpos_idx]
                data.ctrl[act_idx] = 5.0 * kps[act_idx] * error - 3.0 * kds[act_idx] * data.qvel[qvel_idx]
        
        # 시뮬레이션 단계
        mujoco.mj_step(model, data)
        
        # 주기적으로 뷰어 업데이트
        if i % 100 == 0:
            viewer_obj.sync()
            print(f"Stabilization step {i}: height = {data.qpos[2]:.3f}m")
    
    # 고정 해제 후 추가 안정화
    print("\nReleasing base constraints gradually...")
    
    # 점진적 해제를 위한 가중치
    for i in range(500):
        # 가중치 계산 (0 -> 1로 점진적 증가)
        weight = min(1.0, i / 250.0)
        
        # 위치와 방향 부분적으로 제약
        if i < 250:  # 처음 250단계는 천천히 제약 해제
            # 위치 부분 고정
            data.qpos[:3] = (1.0 - weight) * initial_pos + weight * data.qpos[:3]
            
            # 방향 부분 고정 (쿼터니언 선형 보간은 정확하진 않지만 간단한 구현으로 사용)
            data.qpos[3:7] = (1.0 - weight) * initial_quat + weight * data.qpos[3:7]
            # 정규화
            data.qpos[3:7] = data.qpos[3:7] / np.linalg.norm(data.qpos[3:7])
        
        # PD 제어 적용
        for act_idx, joint_info in actuator_to_joint_map.items():
            if act_idx < len(default_angles):
                qpos_idx = joint_info['qpos_idx']
                qvel_idx = joint_info['qvel_idx']
                
                error = default_angles[act_idx] - data.qpos[qpos_idx]
                data.ctrl[act_idx] = 3.0 * kps[act_idx] * error - 2.0 * kds[act_idx] * data.qvel[qvel_idx]
        
        # 시뮬레이션 단계
        mujoco.mj_step(model, data)
        
        # 주기적으로 뷰어 업데이트
        if i % 50 == 0:
            viewer_obj.sync()
            print(f"Free stabilization step {i}: height = {data.qpos[2]:.3f}m, weight = {weight:.2f}")
    
    print("\nStabilization complete. Starting main simulation...")
    
    # 메인 시뮬레이션 루프
    t0 = time.time()
    sim_time = 0.0
    step_counter = 0
    reset_counter = 0
    
    try:
        while sim_time < 30.0 and viewer_obj.is_running() and reset_counter < 5:
            # PD 제어 적용하여 자세 유지
            for act_idx, joint_info in actuator_to_joint_map.items():
                if act_idx < len(default_angles):
                    qpos_idx = joint_info['qpos_idx']
                    qvel_idx = joint_info['qvel_idx']
                    
                    error = default_angles[act_idx] - data.qpos[qpos_idx]
                    data.ctrl[act_idx] = kps[act_idx] * error - kds[act_idx] * data.qvel[qvel_idx]
            
            # 시뮬레이션 진행
            mujoco.mj_step(model, data)
            viewer_obj.sync()
            
            # 시간 및 카운터 업데이트
            sim_time += model.opt.timestep
            step_counter += 1
            
            # 초마다 상태 출력
            if step_counter % int(1.0 / model.opt.timestep) == 0:
                base_height = data.qpos[2]
                base_orientation = data.qpos[3:7]
                print(f"Time: {sim_time:.1f}s, Height: {base_height:.3f}m, Orientation: {base_orientation}")
                
                # 관절 오차 확인
                large_error_count = 0
                for act_idx, joint_info in actuator_to_joint_map.items():
                    if act_idx < len(default_angles):
                        qpos_idx = joint_info['qpos_idx']
                        actual = data.qpos[qpos_idx]
                        target = default_angles[act_idx]
                        error = target - actual
                        if abs(error) > 0.3:  # 0.3 라디안 (약 17도) 이상 차이
                            print(f"  Joint {joint_info['joint_name']}: target={target:.3f}, actual={actual:.3f}, error={error:.3f}")
                            large_error_count += 1
                
                # 넘어졌는지 확인
                if base_height < 0.25 and sim_time > 1.0:  # 1초 이후에만 리셋 적용
                    print(f"Robot appears to have fallen! Height: {base_height:.3f}m")
                    print("Attempting to reset position...")
                    
                    # 위치 및 방향 리셋
                    data.qpos[0:3] = [0, 0, initial_height]
                    data.qpos[3:7] = quaternion_from_euler(0, pitch_angle, 0)  # 약간 뒤로 기울인 자세로 리셋
                    
                    # 속도 초기화
                    data.qvel[:] = 0.0
                    
                    # 관절 위치 리셋
                    for act_idx, joint_info in actuator_to_joint_map.items():
                        if act_idx < len(default_angles):
                            qpos_idx = joint_info['qpos_idx']
                            data.qpos[qpos_idx] = default_angles[act_idx]
                    
                    # 짧은 안정화 단계
                    for i in range(200):
                        # 위치와 방향 고정
                        data.qpos[:3] = [0, 0, initial_height]
                        data.qpos[3:7] = quaternion_from_euler(0, pitch_angle, 0)
                        data.qvel[:6] = 0.0
                        
                        # 강한 PD 제어
                        for act_idx, joint_info in actuator_to_joint_map.items():
                            if act_idx < len(default_angles):
                                qpos_idx = joint_info['qpos_idx']
                                qvel_idx = joint_info['qvel_idx']
                                error = default_angles[act_idx] - data.qpos[qpos_idx]
                                data.ctrl[act_idx] = 5.0 * kps[act_idx] * error - 3.0 * kds[act_idx] * data.qvel[qvel_idx]
                        
                        mujoco.mj_step(model, data)
                        if i % 20 == 0:
                            viewer_obj.sync()
                    
                    reset_counter += 1
                    print(f"Reset complete. New height: {data.qpos[2]:.3f}m")
                
    except KeyboardInterrupt:
        print("Simulation interrupted by user")
    except Exception as e:
        print(f"Error during simulation: {e}")
    
    t1 = time.time()
    print(f"\nSimulation finished. Wall time: {t1-t0:.2f}s, Sim time: {sim_time:.1f}s, Steps: {step_counter}")
    viewer_obj.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', type=str, help='Configuration file')
    args = parser.parse_args()
    main(args.cfg)