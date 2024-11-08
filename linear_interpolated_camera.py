import numpy as np
from scipy.spatial.transform import Rotation as R

#쿼터니언 형태로 입력 받아서, 보간된 카메라 포즈의 쿼터니언을 출력
def slerp(q1, q2, t):
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    dot = np.dot(q1, q2)
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        result = result / np.linalg.norm(result)
        return result
    
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    q3 = q2 - q1 * dot
    q3 = q3 / np.linalg.norm(q3)
    q_interpolated = q1 * np.cos(theta) + q3 * np.sin(theta)
    return q_interpolated

#카메라 2개가 바라보는 중심 찾는 함수
def calculate_center_from_cameras(extrinsic_matrix1, extrinsic_matrix2):
    cam_pos1 = extrinsic_matrix1[:3, 3]
    cam_pos2 = extrinsic_matrix2[:3, 3]
    
    direction1 = extrinsic_matrix1[:3, 2]
    direction2 = extrinsic_matrix2[:3, 2]
    
    direction1 = direction1 / np.linalg.norm(direction1)
    direction2 = direction2 / np.linalg.norm(direction2)
    
    w0 = cam_pos1 - cam_pos2
    a = np.dot(direction1, direction1)
    b = np.dot(direction1, direction2)
    c = np.dot(direction2, direction2)
    d = np.dot(direction1, w0)
    e = np.dot(direction2, w0)
    
    denominator = a * c - b * b
    if abs(denominator) < 1e-6:
        center = (cam_pos1 + cam_pos2) / 2
    else:
        s = (b * e - c * d) / denominator
        t = (a * e - b * d) / denominator
        closest_point1 = cam_pos1 + s * direction1
        closest_point2 = cam_pos2 + t * direction2
        center = (closest_point1 + closest_point2) / 2
    
    return center

#선형보간된 새로운 카메라 포즈들을 4x4 matrix로 바꾸고 save_path에 npy파일로 저장

def interpolate_camera_poses(extrinsic_matrix1, extrinsic_matrix2, num_interpolations, save_path):
    center = calculate_center_from_cameras(extrinsic_matrix1, extrinsic_matrix2)
    cam_pos1 = extrinsic_matrix1[:3, 3]
    cam_pos2 = extrinsic_matrix2[:3, 3]
    distance = np.linalg.norm(cam_pos1 - center)  # 카메라와 중심점 사이의 거리
    
    R1 = extrinsic_matrix1[:3, :3]
    R2 = extrinsic_matrix2[:3, :3]
    
    q1 = R.from_matrix(R1).as_quat()
    q2 = R.from_matrix(R2).as_quat()
    
    for i in range(num_interpolations):
        alpha = i / (num_interpolations - 1)
        
        interp_pos = (1 - alpha) * cam_pos1 + alpha * cam_pos2
        direction = (interp_pos - center) / np.linalg.norm(interp_pos - center)
        
        # 일정한 거리 유지
        interp_pos = center + direction * distance
        
        q_interpolated = slerp(q1, q2, alpha)
        interpolated_R = R.from_quat(q_interpolated).as_matrix()
        
        interpolated_matrix = np.eye(4)
        interpolated_matrix[:3, :3] = interpolated_R
        interpolated_matrix[:3, 3] = interp_pos

        #보간한 카메라는 한번에 저장해야 하면, 이부분 for문 바깥으로 수정필요

        np.save(f"{save_path}/{i}.npy", interpolated_matrix)
        print(f"Interpolated Matrix {i + 1} saved at {save_path}/{i}.npy")


#추후에 여기 입력에다가 poses_bounded.npy나 더스터에서 뽑은 4x4 matrix 주소 넣으면됨
#쿼터니언 형태라면 바꿔서 넣어줄 필요 있음
R1 = np.load("/intern2/bgd/Yaicon/CameraViewer/inputs/quick/treehill/poses/_DSC8904.npy")
R2 = np.load("/intern2/bgd/Yaicon/CameraViewer/inputs/quick/treehill/poses/_DSC8948.npy")

save_path = "/intern2/bgd/Yaicon/CameraViewer/inputs/quick/treehill/poses"
#보간할 카메라 숫자들 
num_interpolations = 30
interpolate_camera_poses(R1, R2, num_interpolations, save_path)

#최종적으로 조금 수정해서, 노션에 나와있는 대로 images.bin 파일로 구성하면 될 듯
