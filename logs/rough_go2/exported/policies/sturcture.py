import torch
import numpy as np
import sys

# 정책 파일 경로
policy_path = sys.argv[1] if len(sys.argv) > 1 else "policy_1.pt"

try:
    # 정책 모델 로드
    policy = torch.jit.load(policy_path)
    
    # 모델 구조 출력
    print("\n=== 모델 구조 ===")
    print(policy)
    
    # 모델 매개변수 출력
    print("\n=== 모델 매개변수 ===")
    total_params = 0
    for name, param in policy.named_parameters():
        print(f"{name}: {param.shape}, 통계 (min: {param.min().item():.3f}, max: {param.max().item():.3f}, mean: {param.mean().item():.3f}, std: {param.std().item():.3f})")
        total_params += param.numel()
    print(f"총 매개변수 수: {total_params:,}")
    
    # 더미 입력 생성 및 테스트
    print("\n=== 더미 입력 테스트 ===")
    # 다양한 입력 차원 시도
    for obs_dim in [47, 48, 49]:
        dummy_input = torch.zeros((1, obs_dim), dtype=torch.float32)
        try:
            # 모든 입력을 1로 설정
            dummy_input_ones = torch.ones((1, obs_dim), dtype=torch.float32)
            output_ones = policy(dummy_input_ones)
            print(f"입력 차원 {obs_dim}(ones): 출력 형태 {output_ones.shape}, 출력 범위 {output_ones.min().item():.3f} ~ {output_ones.max().item():.3f}")
            
            # 정규 분포 난수 생성
            dummy_input_rand = torch.randn((1, obs_dim), dtype=torch.float32)
            output_rand = policy(dummy_input_rand)
            print(f"입력 차원 {obs_dim}(random): 출력 형태 {output_rand.shape}, 출력 범위 {output_rand.min().item():.3f} ~ {output_rand.max().item():.3f}")
            
            # 실제 GO2 관찰 형태로 만들기
            # 기본 템플릿 관찰 생성
            dummy_obs = torch.zeros((1, obs_dim), dtype=torch.float32)
            # base_lin_vel (3) - 로봇이 정지해 있다고 가정
            dummy_obs[0, 0:3] = torch.tensor([0.0, 0.0, 0.0])
            # base_ang_vel (3) - 로봇이 회전하지 않는다고 가정
            dummy_obs[0, 3:6] = torch.tensor([0.0, 0.0, 0.0])
            # gravity (3) - 표준 중력 벡터
            dummy_obs[0, 6:9] = torch.tensor([0.0, 0.0, -1.0])
            # command (3) - 정지 명령
            dummy_obs[0, 9:12] = torch.tensor([0.0, 0.0, 0.0])
            # 나머지는 0으로 채움 (joint positions, velocities, previous actions)
            
            output_template = policy(dummy_obs)
            print(f"현실적인 정지 상태 입력 (차원 {obs_dim}): 출력 형태 {output_template.shape}, 출력 범위 {output_template.min().item():.3f} ~ {output_template.max().item():.3f}")
            print(f"출력 값: {output_template.detach().numpy().squeeze()}")
            
        except Exception as e:
            print(f"입력 차원 {obs_dim}에서 오류 발생: {e}")
    
    # 모델이 LSTM 또는 RNN을 사용하는지 확인
    if hasattr(policy, 'reset_memory'):
        print("\n이 모델은 LSTM/RNN 메모리를 사용합니다.")
        policy.reset_memory()
        dummy_input = torch.zeros((1, 48), dtype=torch.float32)
        output1 = policy(dummy_input)
        print(f"첫 번째 실행 결과: {output1.min().item():.3f} ~ {output1.max().item():.3f}")
        output2 = policy(dummy_input)
        print(f"두 번째 실행 결과: {output2.min().item():.3f} ~ {output2.max().item():.3f}")
        print(f"두 출력의 차이: {torch.norm(output2 - output1).item():.6f}")
    
except Exception as e:
    print(f"모델 로드 또는 분석 중 오류 발생: {e}")