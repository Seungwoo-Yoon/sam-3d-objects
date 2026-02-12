# Dual Backbone Model Usage Guide

## Overview

`DualBackboneSparseStructureFlowTdfyWrapper`를 사용하면 `SparseStructureFlowTdfyWrapper`와 `GlobalSparseStructureFlowTdfyWrapper` 두 개의 백본 모델을 병렬로 실행하고 출력을 결합할 수 있습니다.

### 주요 특징

- **병렬 실행**: 두 모델이 동시에 실행되어 동일한 입력을 처리
- **출력 결합**: 두 모델의 출력을 element-wise addition으로 결합
- **별도 체크포인트 관리**: 각 백본의 체크포인트를 독립적으로 로드/저장
- **단계별 학습**: SparseStructureFlow는 기존 체크포인트 사용, GlobalSparseStructureFlow는 새로 초기화하여 학습 가능

## 파일 구조

```
sam3d_objects/model/backbone/tdfy_dit/models/
├── dual_backbone_sparse_structure_flow.py      # 메인 모델 클래스
├── dual_backbone_checkpoint_utils.py           # 체크포인트 로딩/저장 유틸리티
├── mot_sparse_structure_flow.py                # SparseStructureFlowTdfyWrapper
└── global_sparse_structure_flow.py             # GlobalSparseStructureFlowTdfyWrapper

checkpoints/hf/
├── ss_generator.yaml                           # 기존 단일 백본 설정
└── dual_backbone_generator.yaml                # 새로운 듀얼 백본 설정
```

## 사용 방법

### 1. YAML 설정 파일 사용

`dual_backbone_generator.yaml`을 사용하여 모델을 생성:

```yaml
backbone:
  _target_: sam3d_objects.model.backbone.tdfy_dit.models.dual_backbone_sparse_structure_flow.DualBackboneSparseStructureFlowTdfyWrapper

  # SparseStructureFlow 설정
  sparse_flow_config:
    cond_channels: 1024
    model_channels: 1024
    num_blocks: 24
    # ... 기타 설정

  # GlobalSparseStructureFlow 설정
  global_sparse_flow_config:
    cond_channels: 1024
    model_channels: 1024
    num_blocks: 24
    # ... 기타 설정

  # 체크포인트 경로
  sparse_flow_checkpoint: "path/to/sparse_flow.ckpt"  # 기존 체크포인트
  global_sparse_flow_checkpoint: null  # null이면 새로 초기화

  # 결합 방식
  combine_mode: add  # 'add' 또는 'weighted'
  global_flow_weight: 1.0  # weighted 모드일 때만 사용
```

### 2. Python 코드로 직접 사용

#### 2.1 기본 사용법

```python
from sam3d_objects.model.backbone.tdfy_dit.models import (
    DualBackboneSparseStructureFlowTdfyWrapper,
)

# 각 백본의 설정
sparse_flow_config = {
    "cond_channels": 1024,
    "model_channels": 1024,
    "num_blocks": 24,
    "latent_mapping": {...},
    # ... 기타 설정
}

global_sparse_flow_config = {
    "cond_channels": 1024,
    "model_channels": 1024,
    "num_blocks": 24,
    "latent_mapping": {...},
    # ... 기타 설정
}

# 모델 생성
model = DualBackboneSparseStructureFlowTdfyWrapper(
    sparse_flow_config=sparse_flow_config,
    global_sparse_flow_config=global_sparse_flow_config,
    sparse_flow_checkpoint="checkpoints/sparse_flow.ckpt",  # 기존 체크포인트 로드
    global_sparse_flow_checkpoint=None,  # 새로 초기화
    combine_mode="add",
)
```

#### 2.2 체크포인트 로딩 유틸리티 사용

```python
from sam3d_objects.model.backbone.tdfy_dit.models.dual_backbone_checkpoint_utils import (
    load_dual_backbone_from_checkpoints,
    create_dual_backbone_from_single_checkpoint,
)

# 방법 1: 모델 생성 후 체크포인트 로드
model = DualBackboneSparseStructureFlowTdfyWrapper(
    sparse_flow_config=sparse_flow_config,
    global_sparse_flow_config=global_sparse_flow_config,
)

load_dual_backbone_from_checkpoints(
    model,
    sparse_flow_checkpoint="checkpoints/sparse_flow.ckpt",
    global_sparse_flow_checkpoint=None,  # 없으면 초기화 상태 유지
)

# 방법 2: 기존 단일 체크포인트에서 듀얼 백본 생성
model = create_dual_backbone_from_single_checkpoint(
    checkpoint_path="checkpoints/ss_generator.ckpt",
    dual_backbone_config={
        "sparse_flow_config": sparse_flow_config,
        "global_sparse_flow_config": global_sparse_flow_config,
        "combine_mode": "add",
    },
)
```

### 3. Forward Pass

```python
# 입력 데이터 준비
latents_dict = {
    "shape": shape_latent,
    "6drotation_normalized": rotation_latent,
    "translation": translation_latent,
    "scale": scale_latent,
    "translation_scale": translation_scale_latent,
}

# Forward pass
output = model(
    latents_dict=latents_dict,
    t=timesteps,
    condition_arg1,
    condition_arg2,
    cfg=False,
)

# 출력은 두 백본의 결과를 element-wise로 더한 값
# output = sparse_flow_output + global_sparse_flow_output
```

### 4. 체크포인트 저장

```python
from sam3d_objects.model.backbone.tdfy_dit.models.dual_backbone_checkpoint_utils import (
    save_dual_backbone_checkpoint,
)

# 모든 체크포인트 저장 (별도 파일로)
save_dual_backbone_checkpoint(
    model,
    save_path="checkpoints/dual_backbone/",  # 디렉토리로 저장
    additional_state={"epoch": 10, "step": 1000},
)
# 생성 파일:
# - checkpoints/dual_backbone/sparse_flow.ckpt
# - checkpoints/dual_backbone/global_sparse_flow.ckpt
# - checkpoints/dual_backbone/combined_model.ckpt

# GlobalSparseStructureFlow만 저장
save_dual_backbone_checkpoint(
    model,
    save_path="checkpoints/global_sparse_flow.ckpt",
    save_global_flow_only=True,
)
```

## 학습 워크플로우

### 초기 학습 (GlobalSparseStructureFlow를 새로 학습)

1. **기존 SparseStructureFlow 체크포인트 준비**
   - 기존 `ss_generator.ckpt` 또는 학습된 체크포인트 준비

2. **Dual Backbone 모델 생성**
   ```python
   model = DualBackboneSparseStructureFlowTdfyWrapper(
       sparse_flow_config=sparse_flow_config,
       global_sparse_flow_config=global_sparse_flow_config,
       sparse_flow_checkpoint="checkpoints/ss_generator.ckpt",
       global_sparse_flow_checkpoint=None,  # 새로 초기화
   )
   ```

3. **SparseStructureFlow 가중치 고정 (옵션)**
   ```python
   # SparseStructureFlow는 학습하지 않고 GlobalSparseStructureFlow만 학습
   for param in model.sparse_flow.parameters():
       param.requires_grad = False
   ```

4. **학습**
   ```python
   # 일반적인 학습 루프
   optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

   for epoch in range(num_epochs):
       for batch in dataloader:
           output = model(batch["latents"], batch["t"], ...)
           loss = criterion(output, batch["target"])
           loss.backward()
           optimizer.step()
   ```

5. **체크포인트 저장**
   ```python
   # GlobalSparseStructureFlow만 저장
   save_dual_backbone_checkpoint(
       model,
       save_path=f"checkpoints/global_flow_epoch_{epoch}.ckpt",
       save_global_flow_only=True,
   )
   ```

### 추가 학습 (이미 학습된 체크포인트 사용)

```python
# 두 백본 모두 체크포인트에서 로드
model = DualBackboneSparseStructureFlowTdfyWrapper(
    sparse_flow_config=sparse_flow_config,
    global_sparse_flow_config=global_sparse_flow_config,
    sparse_flow_checkpoint="checkpoints/sparse_flow.ckpt",
    global_sparse_flow_checkpoint="checkpoints/global_flow_epoch_50.ckpt",
)

# 필요에 따라 특정 백본만 학습
# 예: SparseStructureFlow 고정, GlobalSparseStructureFlow만 fine-tuning
for param in model.sparse_flow.parameters():
    param.requires_grad = False
```

## 추론

```python
model.eval()

with torch.no_grad():
    output = model(latents_dict, t, condition_args)

# 출력은 두 백본의 결합된 결과
```

## 주요 매개변수

### DualBackboneSparseStructureFlowTdfyWrapper

| 매개변수 | 타입 | 설명 |
|---------|------|------|
| `sparse_flow_config` | dict | SparseStructureFlowTdfyWrapper 설정 |
| `global_sparse_flow_config` | dict | GlobalSparseStructureFlowTdfyWrapper 설정 |
| `sparse_flow_checkpoint` | str or None | SparseStructureFlow 체크포인트 경로 |
| `global_sparse_flow_checkpoint` | str or None | GlobalSparseStructureFlow 체크포인트 경로 |
| `combine_mode` | 'add' or 'weighted' | 출력 결합 방식 |
| `global_flow_weight` | float | weighted 모드일 때 GlobalFlow의 가중치 |

### combine_mode

- **'add'**: 단순 덧셈 (`output = sparse_output + global_output`)
- **'weighted'**: 학습 가능한 가중치 사용 (`output = sparse_output + α * global_output`)
  - `α`는 학습 중 자동으로 조정됨

## 문제 해결

### 체크포인트 키 불일치

체크포인트 로딩 시 키 불일치 오류가 발생하면:

```python
# strict=False로 설정
load_dual_backbone_from_checkpoints(
    model,
    sparse_flow_checkpoint="path/to/checkpoint.ckpt",
    sparse_flow_strict=False,
)
```

### 메모리 부족

두 개의 큰 모델을 동시에 실행하면 메모리가 부족할 수 있습니다:

1. **Gradient checkpointing 사용**
   ```python
   sparse_flow_config["use_checkpoint"] = True
   global_sparse_flow_config["use_checkpoint"] = True
   ```

2. **FP16 사용**
   ```python
   sparse_flow_config["use_fp16"] = True
   global_sparse_flow_config["use_fp16"] = True
   ```

3. **배치 크기 줄이기**

## 예제 스크립트

전체 예제는 다음 파일들을 참고하세요:

- `train_shortcut_scene.py`: 기본 학습 스크립트
- `train_shortcut_scene_example.py`: 예제 학습 스크립트
- `dual_backbone_generator.yaml`: 설정 파일 예제

## 추가 정보

- 두 백본은 동일한 `latent_mapping` 구조를 가져야 합니다
- 출력 딕셔너리의 키가 일치해야 합니다
- 각 백본은 독립적으로 학습 가능합니다

## 참고

- [mot_sparse_structure_flow.py](sam3d_objects/model/backbone/tdfy_dit/models/mot_sparse_structure_flow.py) - SparseStructureFlow 구현
- [global_sparse_structure_flow.py](sam3d_objects/model/backbone/tdfy_dit/models/global_sparse_structure_flow.py) - GlobalSparseStructureFlow 구현
- [dual_backbone_sparse_structure_flow.py](sam3d_objects/model/backbone/tdfy_dit/models/dual_backbone_sparse_structure_flow.py) - Dual Backbone 구현
