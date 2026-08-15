"""
Module hình học caliper — trục cốt lõi của công cụ.

Caliper lâm sàng thật (như trong ảnh Philips mẫu) gồm 2 cặp điểm tạo thành
2 đoạn thẳng VUÔNG GÓC nhau, không nhất thiết song song trục x/y ảnh:
    - Cặp A (trục dài / "Dist 1"): đo đường kính lớn nhất của khối
    - Cặp B (trục ngắn / "Dist 2"): đo bề rộng vuông góc với trục dài

Module này xử lý 2 CHIỀU:
    1. suggest_calipers_from_mask / suggest_calipers_from_bbox:
       Từ kết quả AI detect (mask hoặc bbox) -> gợi ý 4 điểm caliper ban đầu
       cho bác sĩ, bám theo đúng quy ước lâm sàng (trục dài nhất trước).

    2. lesion_region_from_calipers:
       Từ 4 điểm caliper SAU KHI bác sĩ đã xác nhận/chỉnh sửa -> tính lại
       vùng tổn thương (oriented rect + axis-aligned bbox tương đương) để
       lưu làm nhãn training cho lần retrain sau — đây là dữ liệu "vàng"
       vì được người thật xác nhận trên ảnh gốc 100%, không qua AI inpaint.
"""

import math
from dataclasses import dataclass, asdict

import cv2
import numpy as np


@dataclass
class Point:
    x: float
    y: float

    def as_tuple(self):
        return (self.x, self.y)


@dataclass
class CaliperPair:
    """1 trục đo — 2 điểm caliper cùng loại (2 dấu '+' hoặc 2 dấu 'x')."""
    p1: Point
    p2: Point

    def length(self) -> float:
        return math.hypot(self.p2.x - self.p1.x, self.p2.y - self.p1.y)

    def angle_rad(self) -> float:
        return math.atan2(self.p2.y - self.p1.y, self.p2.x - self.p1.x)

    def midpoint(self) -> Point:
        return Point((self.p1.x + self.p2.x) / 2, (self.p1.y + self.p2.y) / 2)


@dataclass
class OrientedRegion:
    """Vùng tổn thương dạng hình chữ nhật xoay (khớp đúng hướng khối u thật)."""
    center_x: float
    center_y: float
    width: float   # chiều theo trục dài (Dist 1)
    height: float  # chiều theo trục ngắn (Dist 2)
    angle_deg: float  # góc xoay của trục dài so với trục x ảnh
    corners: list  # 4 góc [(x,y), ...] theo thứ tự, để vẽ polygon
    axis_aligned_bbox: dict  # {x1,y1,x2,y2} - bbox thẳng trục tương đương (cho nhãn YOLO detect chuẩn)

    def to_dict(self):
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# CHIỀU 1: AI detect (mask hoặc bbox) -> gợi ý 4 điểm caliper
# ---------------------------------------------------------------------------

def suggest_calipers_from_mask(mask: np.ndarray) -> tuple[CaliperPair, CaliperPair]:
    """
    mask: ảnh nhị phân (H,W), giá trị >0 là vùng tổn thương AI detect được.
    Dùng cv2.minAreaRect để tìm hình chữ nhật xoay khít nhất bao quanh mask
    -> đúng hướng trục dài thật của khối, thay vì áp thẳng trục x/y.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Mask rỗng, không tìm được contour nào để gợi ý caliper.")
    largest = max(contours, key=cv2.contourArea)
    (cx, cy), (w, h), angle_deg = cv2.minAreaRect(largest)
    return _pairs_from_oriented_rect(cx, cy, w, h, angle_deg)


def suggest_calipers_from_bbox(x1: float, y1: float, x2: float, y2: float) -> tuple[CaliperPair, CaliperPair]:
    """
    Fallback khi model chỉ có bbox thẳng trục (không có mask/segmentation).
    Không biết hướng thật của khối -> đặt caliper theo đúng trục ảnh (0 độ),
    cạnh dài hơn của bbox trở thành trục A. Đây chỉ là điểm khởi đầu để bác sĩ
    kéo chỉnh lại cho khớp hướng thật, KHÔNG thay thế được mask thật.
    """
    w = x2 - x1
    h = y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    angle_deg = 0.0 if w >= h else 90.0
    long_side, short_side = (w, h) if w >= h else (h, w)
    return _pairs_from_oriented_rect(cx, cy, long_side, short_side, angle_deg)


def _pairs_from_oriented_rect(cx, cy, w, h, angle_deg) -> tuple[CaliperPair, CaliperPair]:
    """cv2.minAreaRect trả (w,h) không đảm bảo w là cạnh dài -> tự chuẩn hoá
    để trục A (pair A) luôn là trục DÀI hơn, đúng quy ước lâm sàng (Dist 1 > Dist 2)."""
    if w < h:
        w, h = h, w
        angle_deg += 90.0

    theta = math.radians(angle_deg)
    dir_long = (math.cos(theta), math.sin(theta))
    dir_short = (-math.sin(theta), math.cos(theta))

    half_w, half_h = w / 2, h / 2

    pair_a = CaliperPair(
        p1=Point(cx - dir_long[0] * half_w, cy - dir_long[1] * half_w),
        p2=Point(cx + dir_long[0] * half_w, cy + dir_long[1] * half_w),
    )
    pair_b = CaliperPair(
        p1=Point(cx - dir_short[0] * half_h, cy - dir_short[1] * half_h),
        p2=Point(cx + dir_short[0] * half_h, cy + dir_short[1] * half_h),
    )
    return pair_a, pair_b


# ---------------------------------------------------------------------------
# CHIỀU 2: Caliper bác sĩ đã xác nhận -> vùng tổn thương (để lưu dataset)
# ---------------------------------------------------------------------------

def lesion_region_from_calipers(pair_a: CaliperPair, pair_b: CaliperPair) -> OrientedRegion:
    """
    pair_a: trục dài (2 điểm), pair_b: trục ngắn (2 điểm) — do bác sĩ đặt/chỉnh.
    Không giả định 2 trục giao nhau chính xác tại 1 điểm (tay người kéo thả
    hiếm khi tuyệt đối chính xác) -> center lấy theo giao điểm 2 đường thẳng
    nếu tính được, fallback về trung bình 4 điểm nếu 2 đường gần song song
    (trường hợp lỗi/degenerate).
    """
    center = _line_intersection(pair_a, pair_b)
    if center is None:
        all_pts = [pair_a.p1, pair_a.p2, pair_b.p1, pair_b.p2]
        center = Point(sum(p.x for p in all_pts) / 4, sum(p.y for p in all_pts) / 4)

    width = pair_a.length()   # trục dài
    height = pair_b.length()  # trục ngắn
    angle_deg = math.degrees(pair_a.angle_rad())

    theta = math.radians(angle_deg)
    dir_long = (math.cos(theta), math.sin(theta))
    dir_short = (-math.sin(theta), math.cos(theta))
    half_w, half_h = width / 2, height / 2

    corners = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            cx_ = center.x + sx * dir_long[0] * half_w + sy * dir_short[0] * half_h
            cy_ = center.y + sx * dir_long[1] * half_w + sy * dir_short[1] * half_h
            corners.append((cx_, cy_))
    # Sắp lại theo thứ tự polygon hợp lệ (không tự cắt chéo): dùng đúng thứ tự
    # sinh ra ở trên đã là 4 góc liên tiếp hợp lệ của hình chữ nhật (sx,sy) = (-,-),(-,+),(+,+),(+,-)
    corners = [corners[0], corners[1], corners[3], corners[2]]

    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    bbox = {"x1": min(xs), "y1": min(ys), "x2": max(xs), "y2": max(ys)}

    return OrientedRegion(
        center_x=center.x, center_y=center.y,
        width=width, height=height, angle_deg=angle_deg,
        corners=corners, axis_aligned_bbox=bbox,
    )


def _line_intersection(pair_a: CaliperPair, pair_b: CaliperPair):
    x1, y1, x2, y2 = pair_a.p1.x, pair_a.p1.y, pair_a.p2.x, pair_a.p2.y
    x3, y3, x4, y4 = pair_b.p1.x, pair_b.p1.y, pair_b.p2.x, pair_b.p2.y
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-9:
        return None  # gần song song, không giao nhau đáng tin cậy
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return Point(px, py)


def bbox_iou(a: dict, b: dict) -> float:
    """IoU giữa 2 axis-aligned bbox dạng {x1,y1,x2,y2}."""
    ix1, iy1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    ix2, iy2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    area_b = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def to_yolo_obb_label(region: OrientedRegion, img_w: int, img_h: int, class_id: int = 0) -> str:
    """Xuất nhãn theo format YOLO OBB (Ultralytics): class x1 y1 x2 y2 x3 y3 x4 y4 (normalized 0-1)."""
    coords = []
    for (x, y) in region.corners:
        coords.append(f"{x / img_w:.6f}")
        coords.append(f"{y / img_h:.6f}")
    return f"{class_id} " + " ".join(coords)


def to_yolo_bbox_label(region: OrientedRegion, img_w: int, img_h: int, class_id: int = 0) -> str:
    """Xuất nhãn theo format YOLO detect chuẩn (axis-aligned): class cx cy w h (normalized 0-1)."""
    bbox = region.axis_aligned_bbox
    cx = (bbox["x1"] + bbox["x2"]) / 2 / img_w
    cy = (bbox["y1"] + bbox["y2"]) / 2 / img_h
    w = (bbox["x2"] - bbox["x1"]) / img_w
    h = (bbox["y2"] - bbox["y1"]) / img_h
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
