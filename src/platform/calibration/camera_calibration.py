import cv2
import numpy as np
import glob
import os
import json
import argparse
from pathlib import Path

def calibrate_camera(
    image_dir: str,
    output_file: str,
    rows: int = 7,
    cols: int = 6,
    square_size: float = 0.030  # meters
):
    """
    Calibrate camera using checkerboard images.
    
    Args:
        image_dir: Directory containing calibration images
        output_file: Path to save calibration JSON
        rows: Number of inner corners per row
        cols: Number of inner corners per column
        square_size: Size of a checkerboard square in meters
    """
    print(f"Calibrating with {rows}x{cols} board, {square_size*1000:.1f}mm squares")
    
    # Termination criteria for sub-pixel refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp = objp * square_size
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    images = glob.glob(os.path.join(image_dir, '*.jpg')) + \
             glob.glob(os.path.join(image_dir, '*.png'))
    
    if not images:
        print(f"No images found in {image_dir}")
        return
        
    print(f"Found {len(images)} images")
    
    valid_images = 0
    image_size = None
    
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if image_size is None:
            image_size = gray.shape[::-1]
        elif gray.shape[::-1] != image_size:
            print(f"Skipping {fname}: inconsistent size")
            continue
            
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            valid_images += 1
            print(f"  [OK] {os.path.basename(fname)}")
        else:
            print(f"  [FAIL] {os.path.basename(fname)}")
            
    if valid_images < 10:
        print("Warning: Less than 10 valid images. Calibration may be poor.")
        
    if valid_images == 0:
        print("Calibration failed: No valid images found.")
        return

    print("Computing calibration...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )
    
    print(f"RMS Error: {ret:.4f}")
    
    # Save results
    data = {
        "camera_id": "calibrated_camera",
        "camera_matrix": mtx.tolist(),
        "dist_coeffs": dist.tolist(),
        "image_size": list(image_size),
        "rms_error": ret,
        "R": np.eye(3).tolist(), # Identity rotation (local frame)
        "t": [0.0, 0.0, 0.0]     # Zero translation (local frame)
    }
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
        
    print(f"Calibration saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera Calibration Tool")
    parser.add_argument("--images", required=True, help="Directory of images")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--rows", type=int, default=7, help="Inner corners rows")
    parser.add_argument("--cols", type=int, default=6, help="Inner corners cols")
    parser.add_argument("--size", type=float, default=0.030, help="Square size (m)")
    
    args = parser.parse_args()
    
    calibrate_camera(
        args.images,
        args.output,
        rows=args.rows,
        cols=args.cols,
        square_size=args.size
    )
