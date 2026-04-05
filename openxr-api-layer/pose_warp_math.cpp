// MIT License
// Pose Pre-Warp Math - Homography Computation Implementation

#include "pch.h"
#include "pose_warp_math.h"
#include <cmath>
#include <algorithm>

namespace pose_warp {

CameraIntrinsics computeIntrinsics(float fovLeft, float fovRight, float fovUp, float fovDown,
                                   int width, int height) {
    // Compute horizontal and vertical FOV from asymmetric FOV angles
    const float fovHorizontal = fovLeft + fovRight;
    const float fovVertical = fovUp + fovDown;
    
    // Compute focal lengths in pixels
    // fx = width / (2 * tan(fovHorizontal / 2))
    const float fx = static_cast<float>(width) / (2.0f * std::tan(fovHorizontal / 2.0f));
    const float fy = static_cast<float>(height) / (2.0f * std::tan(fovVertical / 2.0f));
    
    // Compute principal point (optical center)
    // For asymmetric FOV, principal point is offset from image center
    const float cx = static_cast<float>(width) / 2.0f + fx * std::tan((fovRight - fovLeft) / 2.0f);
    const float cy = static_cast<float>(height) / 2.0f + fy * std::tan((fovDown - fovUp) / 2.0f);
    
    return CameraIntrinsics{fx, fy, cx, cy};
}

void quaternionToMatrix3x3(const XrQuaternionf& q, float R[9]) {
    // Convert unit quaternion to 3x3 rotation matrix
    // Row-major ordering: R[0..2] = row 0, R[3..5] = row 1, R[6..8] = row 2
    
    const float x = q.x;
    const float y = q.y;
    const float z = q.z;
    const float w = q.w;
    
    const float xx = x * x;
    const float xy = x * y;
    const float xz = x * z;
    const float xw = x * w;
    const float yy = y * y;
    const float yz = y * z;
    const float yw = y * w;
    const float zz = z * z;
    const float zw = z * w;
    
    R[0] = 1.0f - 2.0f * (yy + zz);
    R[1] = 2.0f * (xy - zw);
    R[2] = 2.0f * (xz + yw);
    
    R[3] = 2.0f * (xy + zw);
    R[4] = 1.0f - 2.0f * (xx + zz);
    R[5] = 2.0f * (yz - xw);
    
    R[6] = 2.0f * (xz - yw);
    R[7] = 2.0f * (yz + xw);
    R[8] = 1.0f - 2.0f * (xx + yy);
}

void computeRotationHomography(const XrQuaternionf& rotation,
                               const CameraIntrinsics& intrinsics,
                               float homography[9]) {
    // Homography for rotation-only warp: H = K * R * K^-1
    // where K is camera intrinsic matrix, R is rotation matrix
    
    // Step 1: Convert quaternion to rotation matrix
    float R[9];
    quaternionToMatrix3x3(rotation, R);
    
    // Step 2: Build intrinsic matrix K
    // K = [fx  0  cx]
    //     [ 0 fy  cy]
    //     [ 0  0   1]
    const float fx = intrinsics.fx;
    const float fy = intrinsics.fy;
    const float cx = intrinsics.cx;
    const float cy = intrinsics.cy;
    
    // Step 3: Build inverse intrinsic matrix K^-1
    // K^-1 = [1/fx    0   -cx/fx]
    //        [   0  1/fy  -cy/fy]
    //        [   0    0        1]
    const float fx_inv = 1.0f / fx;
    const float fy_inv = 1.0f / fy;
    
    // Step 4: Compute H = K * R * K^-1 via manual matrix multiplication
    // First compute R * K^-1 (store in temp)
    float temp[9];
    
    temp[0] = R[0] * fx_inv;
    temp[1] = R[1] * fy_inv;
    temp[2] = -R[0] * cx * fx_inv - R[1] * cy * fy_inv + R[2];
    
    temp[3] = R[3] * fx_inv;
    temp[4] = R[4] * fy_inv;
    temp[5] = -R[3] * cx * fx_inv - R[4] * cy * fy_inv + R[5];
    
    temp[6] = R[6] * fx_inv;
    temp[7] = R[7] * fy_inv;
    temp[8] = -R[6] * cx * fx_inv - R[7] * cy * fy_inv + R[8];
    
    // Then compute K * temp = H
    homography[0] = fx * temp[0] + cx * temp[6];
    homography[1] = fx * temp[1] + cx * temp[7];
    homography[2] = fx * temp[2] + cx * temp[8];
    
    homography[3] = fy * temp[3] + cy * temp[6];
    homography[4] = fy * temp[4] + cy * temp[7];
    homography[5] = fy * temp[5] + cy * temp[8];
    
    homography[6] = temp[6];
    homography[7] = temp[7];
    homography[8] = temp[8];
}

bool invertHomography(const float H[9], float H_inv[9]) {
    // Compute determinant
    const float det = H[0] * (H[4] * H[8] - H[5] * H[7])
                    - H[1] * (H[3] * H[8] - H[5] * H[6])
                    + H[2] * (H[3] * H[7] - H[4] * H[6]);
    
    // Check for singularity
    constexpr float epsilon = 1e-10f;
    if (std::abs(det) < epsilon) {
        return false;
    }
    
    const float det_inv = 1.0f / det;
    
    // Compute adjugate matrix and multiply by 1/det
    H_inv[0] = (H[4] * H[8] - H[5] * H[7]) * det_inv;
    H_inv[1] = (H[2] * H[7] - H[1] * H[8]) * det_inv;
    H_inv[2] = (H[1] * H[5] - H[2] * H[4]) * det_inv;
    
    H_inv[3] = (H[5] * H[6] - H[3] * H[8]) * det_inv;
    H_inv[4] = (H[0] * H[8] - H[2] * H[6]) * det_inv;
    H_inv[5] = (H[2] * H[3] - H[0] * H[5]) * det_inv;
    
    H_inv[6] = (H[3] * H[7] - H[4] * H[6]) * det_inv;
    H_inv[7] = (H[1] * H[6] - H[0] * H[7]) * det_inv;
    H_inv[8] = (H[0] * H[4] - H[1] * H[3]) * det_inv;
    
    return true;
}

} // namespace pose_warp
