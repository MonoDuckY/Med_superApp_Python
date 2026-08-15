"""
HỆ THỐNG QUẢN LÝ PROMPT CHO LLM GEMINI
Dự án: Med Super App - Medical AI Service
"""

# ==============================================================================
# PROMPT MẪU CỐ ĐỊNH CHO MỖI LẦN CHẠY (USER PROMPT TEMPLATE)
# ==============================================================================
DEFAULT_CALIPER_REMOVAL_PROMPT = """
- Xóa các caliper trong ảnh, giữ nguyên cấu trúc phía bên trong caliper  
- giữ lại toàn bộ cấu trúc của các vùng khác, không chỉnh sửa bất cứ điều gì ở các vùng khác
- giữ nguyên đặc trưng nhiễu hạt của ảnh siêu âm, nhớ rằng giữ nguyên màu sắc, cũng như các cấu trúc khác của các vùng nằm trog các caliper. 
- Giữ nguyên các texture  bên trong các caliper ( không được tự động ngả màu rgb giữ nguyên màu của ảnh siêu âm))
""".strip()


# ==============================================================================
# SYSTEM PROMPT ĐẦY ĐỦ GỬI ĐẾN GOOGLE GEMINI MULTIMODAL VISION API
# ==============================================================================
def build_gemini_caliper_prompt(custom_instruction: str = None) -> str:
    """
    Hàm đóng gói prompt gửi tới Gemini API, kết hợp chỉ thị người dùng và định dạng JSON chuẩn.
    """
    user_rules = custom_instruction if custom_instruction else DEFAULT_CALIPER_REMOVAL_PROMPT
    
    return f"""You are an expert medical ultrasound vision AI assistant.

CRITICAL USER SPECIFICATIONS & RULES:
\"\"\"
{user_rules}
\"\"\"

DETECTION & ANNOTATION INSTRUCTIONS:
1. Scan the ultrasound image and detect ALL synthetic caliper measurement markers overlaid on the scan:
   - Plus sign crosshairs (+)
   - Cross markers (x)
   - Dotted or dashed distance measurement lines
   - Measurement calipers and calibration ticks
   - Distance tags and numerical text labels
2. Follow strict isolation:
   - Detect strictly artificial overlaid caliper markings.
   - Do NOT detect biological organs, cyst boundaries, bone contours, or anatomical structures.
3. Return the detected calipers strictly as a JSON array of bounding boxes with normalized coordinates [ymin, xmin, ymax, xmax] in scale 0 to 1000:
[
  {{
    "box_2d": [ymin, xmin, ymax, xmax],
    "label": "caliper_cross" | "caliper_line" | "caliper_marker",
    "reasoning": "Location and type of caliper marker detected"
  }}
]
"""
