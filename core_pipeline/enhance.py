import cv2
import numpy as np
from numba import jit

@jit(nopython=True, cache=True)
def srad_core(I, n_iter, delta_t, q0):
    rows, cols = I.shape
    for _ in range(n_iter):
        I_next = I.copy()
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                dN = I[r-1, c] - I[r, c]
                dS = I[r+1, c] - I[r, c]
                dW = I[r, c-1] - I[r, c]
                dE = I[r, c+1] - I[r, c]
                
                grad_sq = (dN**2 + dS**2 + dW**2 + dE**2) / (I[r, c]**2 + 1e-5)
                laplacian = (dN + dS + dW + dE) / (I[r, c] + 1e-5)
                
                num = 0.5 * grad_sq - (1.0 / 16.0) * (laplacian**2)
                den = (1.0 + 0.25 * laplacian)**2
                q_sq = num / (den + 1e-5)
                if q_sq < 0: q_sq = 0
                q = np.sqrt(q_sq)
                
                xi_num = q**2 - q0**2
                xi_den = q0**2 * (1.0 + q0**2)
                xi = xi_num / (xi_den + 1e-5)
                
                c_c = 1.0 / (1.0 + xi)
                if c_c > 1.0: c_c = 1.0
                if c_c < 0.0: c_c = 0.0
                
                divergence = c_c*dN + c_c*dS + c_c*dW + c_c*dE
                I_next[r, c] = I[r, c] + (delta_t / 4.0) * divergence
        I = I_next
    return I

def apply_srad(img_rgb: np.ndarray, n_iter: int = 15) -> np.ndarray:
    """
    Áp dụng Speckle Reducing Anisotropic Diffusion (SRAD) để khử nhiễu ảnh siêu âm.
    Dựa trên thuật toán của MinhVB.
    """
    if n_iter == 0:
        return img_rgb
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    delta_t = 0.15
    q0 = 1.0 / np.sqrt(n_iter * delta_t + 1.0)
    
    gray_filtered = srad_core(gray, n_iter, delta_t, q0)
    gray_filtered = np.clip(gray_filtered, 0, 255).astype(np.uint8)
    
    return cv2.cvtColor(gray_filtered, cv2.COLOR_GRAY2RGB)

def adjust_brightness_contrast(image: np.ndarray, brightness: int = 0, contrast: float = 1.0) -> np.ndarray:
    """
    Điều chỉnh độ sáng và độ tương phản.
    - brightness: [-255, 255]
    - contrast: [0.0, 3.0]
    """
    if brightness == 0 and contrast == 1.0:
        return image
    
    # img = img * contrast + brightness
    result = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    return result

def adjust_sharpness(image: np.ndarray, amount: float = 0.0) -> np.ndarray:
    """
    Tăng độ sắc nét bằng Unsharp Masking.
    - amount: mức độ sắc nét [0.0, 2.0]
    """
    if amount <= 0.0:
        return image
        
    blurred = cv2.GaussianBlur(image, (0, 0), 3.0)
    # result = original * (1 + amount) - blurred * amount
    result = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    return result
