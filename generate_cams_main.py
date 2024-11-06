import os
import argparse
import torch
import read_write_model as rw
import numpy as np
from scene.generate_new_cam import gen_new_cam
from scene.dataset_readers import read_extrinsics_text, read_intrinsics_text, readColmapCameras
from scene.cameras import Camera

scene_path = "scenes_sparse/bicycle_large/"
output_path = "output/bicycle_large/"

def find_intersection(cam1, cam2):
    # 두 카메라의 방향 벡터 계산
    dir1 = cam1.R[:, 2]  # 카메라 1의 방향 벡터 (Z 축)
    dir2 = cam2.R[:, 2]  # 카메라 2의 방향 벡터 (Z 축)
    
    # 두 방향 벡터의 시작점 (카메라 위치)
    pos1 = cam1.T.flatten()
    pos2 = cam2.T.flatten()
    
    # 두 직선의 교차점을 찾기 위한 방정식
    A = np.vstack((dir1, -dir2)).T
    b = pos2 - pos1
    t, s = np.linalg.lstsq(A, b, rcond=None)[0]
    
    # 교차점 좌표
    intersection = pos1 + t * dir1
    return intersection

def create_circle_path(cam1, cam2, intersection):
    # 두 카메라 위치
    pos1 = cam1.T.flatten()
    pos2 = cam2.T.flatten()
    
    # 원의 중심과 반지름 계산
    circle_center = intersection
    radius = np.linalg.norm(pos1 - circle_center)
    
    # 두 카메라 위치 벡터로부터 각도 계산
    vec1 = pos1 - circle_center
    vec2 = pos2 - circle_center
    angle_between = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

def generate_cameras_between(cam1, cam2, N, rot_axis='y'):
    intersection = find_intersection(cam1, cam2)
    circle_center, radius, angle_between = create_circle_path(cam1, cam2, intersection)

    angle_step = angle_between / (N + 1)

    new_cams = {}
    for i in range(1, N+1):
        angle = angle_step * i
        
        new_cams[gen_new_cam.colmap_id] = gen_new_cam(cam1, angle, rot_axis)
    return new_cams

def cam_info_to_cam_obj(cam_infos):
    original_cams = {}
    for cam_info in cam_infos:
        cam = Camera(
        colmap_id=cam_info.uid,  # CameraInfo에서 uid를 colmap_id로 사용
        model=cam_info.model,
        R=torch.from_numpy(cam_info.R).to(torch.float32),  # 회전 행렬
        T=torch.from_numpy(cam_info.T).to(torch.float32),  # 변환 벡터
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        image=cam_info.image,
        depth=cam_info.depth,
        gt_alpha_mask=None,  # 필요한 경우 마스크 추가 가능
        image_name=cam_info.image_name,
        uid=cam_info.uid,
        warp_mask=None,  # 초기화 시엔 None으로 설정
        K=torch.eye(4),  # 카메라 내부 행렬
        src_R=None,  # 원본 회전 행렬이 있다면 추가
        src_T=None,  # 원본 변환 행렬이 있다면 추가
        src_uid=cam_info.uid,  # src_uid를 cam_info.uid로 설정
        trans=np.array([0.0, 0.0, 0.0]),  # 기본 변환 설정
        scale=1.0,
        data_device="cuda" 
        )

        original_cams[cam.colmap_id] = cam
    return original_cams

def cam_to_extrinsic_dic(cams):
    images = {}
    for cam in cams.values:
        image_id = cam.colmap_id
        qvec = rw.rotmat2qvec(cam.R)
        tvec = cam.T
        camera_id = cam.colmap_id
        name=cam.image_name
    images[image_id] = rw.Image(
        id=image_id,
        qvec=qvec,
        tvec=tvec,
        camera_id=camera_id,
        name=name,
        xys=None,
        point3D_ids=None
    )
    return images


cameras_extrinsic_file = os.path.join(scene_path, "sparse/0", "images.txt")
cameras_intrinsic_file = os.path.join(scene_path, "sparse/0", "cameras.txt")
cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

cam_infos = readColmapCameras(cam_extrinsics, cam_intrinsics, os.path.join(scene_path, "images"), load_depth=False)


src_cams = cam_info_to_cam_obj(cam_infos)

# src_cams dict의 각 첫번째, 두번째 카메라 obj 반환
cam1 = list(src_cams.values())[0]
cam2 = list(src_cams.values())[1]

# new_cams = generate_cameras_between(cam1, cam2, 3)
# new_images = cam_to_extrinsic_dic(new_cams)

# print(src_cams)
# print(new_cams)

# rw.write_cameras_text(new_cams, os.path.join(output_path, "cameras.txt"))
# rw.write_images_text(new_images, output_path, "images.txt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate interpolated cameras between two given cameras.")
    parser.add_argument('--cam1_id', type=int, required=True, help="COLMAP ID of the first camera.")
    parser.add_argument('--cam2_id', type=int, required=True, help="COLMAP ID of the second camera.")
    parser.add_argument('--num_interpolations', type=int, required=True, help="Number of interpolated cameras to generate.")

    args = parser.parse_args()

    cam1_id = args.cam1_id
    cam2_id = args.cam2_id
    N = args.num_interpolations

    # Ensure that the camera IDs exist in the loaded cameras
    if cam1_id not in src_cams or cam2_id not in src_cams:
        raise ValueError(f"Camera IDs {cam1_id} or {cam2_id} not found in the source cameras.")

    cam1 = src_cams[cam1_id]
    cam2 = src_cams[cam2_id]

    # Generate interpolated cameras
    new_cams = generate_cameras_between(cam1, cam2, N)
    new_images = cam_to_extrinsic_dic(new_cams)

    # Save generated camera data
    os.makedirs(output_path, exist_ok=True)
    rw.write_cameras_text(new_cams, os.path.join(output_path, "cameras.txt"))
    rw.write_images_text(new_images, os.path.join(output_path, "images.txt"))

    print(f"Generated {N} interpolated cameras between camera {cam1_id} and {cam2_id}.")
    print(f"Saved new camera data to {output_path}.")
