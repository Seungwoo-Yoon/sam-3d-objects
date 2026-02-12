# GSO Dataset Download Guide

Google Scanned Objects (GSO) 데이터셋 다운로드 및 설정 가이드입니다.

## GSO 데이터셋 정보

- **공식 페이지**: [Google Research Blog](https://research.google/blog/scanned-objects-by-google-research-a-dataset-of-3d-scanned-common-household-items/)
- **논문**: [arXiv:2204.11918](https://arxiv.org/abs/2204.11918)
- **라이선스**: Creative Commons
- **내용**: 1000개 이상의 고품질 3D 스캔된 일상 생활용품

## 빠른 시작

### 방법 1: 전체 데이터셋 다운로드 (권장)

전체 GSO 데이터셋을 다운로드합니다 (~13GB):

```bash
python download_gso_dataset.py --output-dir ./google_scanned_objects
```

### 방법 2: 필요한 객체만 선택적으로 다운로드

FoundationPose 데이터셋에서 실제로 사용되는 GSO 객체만 다운로드합니다:

```bash
# 1. FoundationPose에서 사용되는 GSO 객체 목록 추출
python extract_gso_objects_list.py \
    --data-root ./foundationpose \
    --output gso_objects.txt

# 2. 해당 객체만 다운로드
python download_gso_dataset.py \
    --output-dir ./google_scanned_objects \
    --objects-list gso_objects.txt
```

## 상세 사용법

### download_gso_dataset.py 옵션

```bash
python download_gso_dataset.py \
    --output-dir ./google_scanned_objects \  # 출력 디렉토리
    --objects-list gso_objects.txt \          # [선택] 특정 객체만 다운로드
    --keep-zip \                              # [선택] ZIP 파일 보관
    --skip-download                           # [선택] 다운로드 건너뛰고 압축 해제만
```

### extract_gso_objects_list.py 옵션

```bash
python extract_gso_objects_list.py \
    --data-root ./foundationpose \  # FoundationPose 데이터 경로
    --output gso_objects.txt        # 출력 파일명
```

## 데이터셋 구조

다운로드 후 구조:

```
google_scanned_objects/
├── Amys_Chunky_Vegetable_Soup/
│   ├── meshes/
│   │   └── model.obj
│   ├── materials/
│   └── thumbnails/
├── adizero_5Tool_25/
│   ├── meshes/
│   │   └── model.obj
│   └── ...
└── ... (1000+ objects)
```

## FoundationPoseDataset과 함께 사용

```python
from foundation_pose_dataset import FoundationPoseDataset

# GSO mesh와 함께 로드
dataset = FoundationPoseDataset(
    data_root='/workspace/sam-3d-objects/foundationpose',
    gso_root='/workspace/sam-3d-objects/google_scanned_objects/google_scanned_objects',
    load_meshes=True,
    mesh_num_samples=2048,
)

# 데이터 로드
sample = dataset[0]

# Mesh 데이터 접근
if 'mesh_data' in sample:
    mesh_points = sample['mesh_data']['mesh_points']  # [N, 2048, 3]
    mesh_available = sample['mesh_data']['mesh_available']  # [N] boolean

    print(f"Objects with meshes: {mesh_available.sum().item()}/{len(mesh_available)}")
```

## 예제: 현재 프로젝트에서 사용

```bash
# 1. FoundationPose 데이터에서 사용되는 GSO 객체 확인
python extract_gso_objects_list.py \
    --data-root /workspace/sam-3d-objects/foundationpose \
    --output /workspace/sam-3d-objects/required_gso_objects.txt

# 2. GSO 데이터 다운로드
python download_gso_dataset.py \
    --output-dir /workspace/sam-3d-objects/gso_dataset \
    --objects-list /workspace/sam-3d-objects/required_gso_objects.txt

# 3. 테스트
python foundation_pose_dataset.py \
    --data-root /workspace/sam-3d-objects/foundationpose \
    --gso-root /workspace/sam-3d-objects/gso_dataset/google_scanned_objects \
    --load-meshes
```

## 트러블슈팅

### 다운로드 실패

다운로드가 실패하면 수동으로 다운로드할 수 있습니다:

1. **INRIA 미러에서 다운로드**:
   ```bash
   wget https://www.paris.inria.fr/archive_ylabbeprojectsdata/megapose/tars/google_scanned_objects.zip
   ```

2. **ZIP 파일을 적절한 위치에 배치**:
   ```bash
   mv google_scanned_objects.zip ./google_scanned_objects/
   ```

3. **압축 해제만 실행**:
   ```bash
   python download_gso_dataset.py \
       --output-dir ./google_scanned_objects \
       --skip-download
   ```

### 디스크 공간 부족

전체 데이터셋은 약 13GB입니다. 공간이 부족하면:
- `--objects-list`를 사용하여 필요한 객체만 다운로드
- 다운로드 후 `--keep-zip` 옵션 없이 실행하여 ZIP 파일 자동 삭제

### Mesh 파일을 찾을 수 없음

`_load_gso_mesh()`는 다음 경로를 자동으로 확인합니다:
- `{object_name}/meshes/model.obj`
- `{object_name}/meshes/model.ply`
- `{object_name}/model.obj`
- `{object_name}/model.ply`
- `{object_name}/{object_name}.obj`
- `{object_name}/{object_name}.ply`

파일 구조가 다르면 foundation_pose_dataset.py의 `_load_gso_mesh()` 함수를 수정하세요.

## 참고 자료

- [Google Scanned Objects 공식 블로그](https://research.google/blog/scanned-objects-by-google-research-a-dataset-of-3d-scanned-common-household-items/)
- [GSO 논문 (arXiv)](https://arxiv.org/abs/2204.11918)
- [Papers with Code - GSO Dataset](https://paperswithcode.com/dataset/google-scanned-objects)

## 라이선스

GSO 데이터셋은 Creative Commons 라이선스 하에 제공됩니다. 사용 시 적절한 출처 표기가 필요합니다.

**Citation**:
```bibtex
@article{downs2022google,
  title={Google scanned objects: A high-quality dataset of 3d scanned household items},
  author={Downs, Laura and Francis, Anthony and Koenig, Nate and Kinman, Brandon and Hickman, Ryan and Reymann, Krista and McHugh, Thomas B and Vanhoucke, Vincent},
  journal={arXiv preprint arXiv:2204.11918},
  year={2022}
}
```
