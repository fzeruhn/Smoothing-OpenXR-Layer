// MIT License
// Pose Pre-Warp Math - Homography Computation for Rotation-Only Warping

#pragma once

#include "pch.h"

namespace pose_warp {

// Camera intrinsics derived from FOV
struct CameraIntrinsics {
    float fx;  // focal length X (pixels)
    float fy;  // focal length Y (pixels)
    float cx;  // principal point X (pixels)
    float cy;  // principal point Y (pixels)
};

// Compute camera intrinsics from OpenXR FOV and image dimensions
// FOV angles are in radians, as provided by XrFovf
CameraIntrinsics computeIntrinsics(float fovLeft, float fovRight, float fovUp, float fovDown,
                                   int width, int height);

// Compute 3x3 homography matrix for rotation-only warp
// rotation: quaternion representing pose delta (display_pose * inverse(render_pose))
// intrinsics: camera parameters from computeIntrinsics
// Output: row-major 3x3 matrix H such that p_warped = H * p_original
void computeRotationHomography(const XrQuaternionf& rotation,
                               const CameraIntrinsics& intrinsics,
                               float homography[9]);

// Helper: Convert quaternion to 3x3 rotation matrix
void quaternionToMatrix3x3(const XrQuaternionf& q, float R[9]);

// Helper: Invert 3x3 homography matrix (needed for backward warp)
// Returns false if matrix is singular
bool invertHomography(const float H[9], float H_inv[9]);

} // namespace pose_warp
