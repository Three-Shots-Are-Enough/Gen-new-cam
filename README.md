# Gen-new-cam

### Environment
* scenes, utils는 SparseGS 폴더를 그대로 가져와서 일부 수정
* torch, CUDA setting은 혹시나 싶어 MVSplat envirnoment와 맞춰둠:
```bash
git clone https://github.com/Three-Shots-Are-Enough/Gen-new-cam
cd Gen-new-cam
conda create -n mvsplat python=3.10
conda activate mvsplat
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
conda install -c conda-forge libgl
```


### Preparation
* `scenes_spare` 폴더는 구글드라이브에 올린 그거라, 내부에 images 폴더가 없어서 만들어주어야 함
* `generate_cams_main.py`가 작업중인 코드이고, 처음에 `scene_path`와 `output_path` 적절히 수정 필요

## Overall Pipeline
실행은 다음 형식으로 구성:
```
python generate_cams_main.py --cam1_id 8686 --cam2_id 8867 --num_interpolations 88
```


현재 코드 구조는
1) 지정 `scene_sparse` 경로에서 `camera.txt`, `images.txt`를 읽기
2) .txt 안의 cam intrinsic, extrinsic 모아서 Camera info 생성 -> Came obj 생성
3) `{colmap_id2: Cam1, colmap_id2: Cam2}` 형태로 `src_cams` 구성
4) src_cams의 [0], [1] 인덱스 카메라와 `num_interpolation` 넘겨받아 num_interpolation 수만큼 scene/generate_new_cam의 gen_new_cam 함수 돌리기
5) 3)과 같은 구조로 new cam만 모은 딕셔너리 반환
6) (`write_images_txt` 인풋 형태 맞추려고) 따로 extrinsic 파라미터만 모아서 images 딕셔너리 만들기
7) COLMAP repo에서 가져온 `read_write_model` 코드의 `write_cameras_txt`, `write_images_txt` 로 `output/scene_name` 경로에 아웃풋 저장
입니다.

## Contribute: Please submit [Issues](https://github.com/Three-Shots-Are-Enough/Gen-new-cam/issues)
 
