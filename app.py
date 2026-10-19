"""
================================================================================
 AI วิเคราะห์ความสามารถบริหารรายรับรายจ่าย
--------------------------------------------------------------------------------
 แอปนี้โหลดโมเดล Linear Regression ที่ฝึกไว้จาก Orange Data Mining (ไฟล์ .pkcls)
 แล้วให้ผู้ใช้กรอกข้อมูลทางการเงินของตนเอง เพื่อทำนาย "คะแนนเครดิต"
 (credit_score: ค่าตัวเลขต่อเนื่อง ช่วงมาตรฐาน 300-850 แบบ FICO/VantageScore)
================================================================================
"""

import os
import joblib
import numpy as np
import streamlit as st
from Orange.data import Domain, Table

# ------------------------------------------------------------------------------
# 1) กำหนดค่าคงที่ / โครงสร้างฟีเจอร์ (schema) ตามชุดข้อมูลที่ใช้ฝึกโมเดล
# ------------------------------------------------------------------------------
MODEL_DIR = "."
TARGET_NAME = "credit_score"
SCORE_MIN, SCORE_MAX = 300, 850

# เกณฑ์แบ่งระดับคะแนน: ปรับคำอธิบายเน้นย้ำว่าเป็นค่าเชิงสาธิต ไม่ใช่การตัดสินสินเชื่อจริง
CREDIT_BANDS = [
    (800, "ดีเยี่ยม (Exceptional)", "ระดับคะแนนจำลองกลุ่มความเสี่ยงต่ำมาก (เพื่อการสาธิต)"),
    (740, "ดีมาก (Very Good)", "ระดับคะแนนจำลองกลุ่มประวัติการเงินดี (เพื่อการสาธิต)"),
    (670, "ดี (Good)", "ระดับคะแนนจำลองกลุ่มเกณฑ์เฉลี่ย (เพื่อการสาธิต)"),
    (580, "พอใช้ (Fair)", "ระดับคะแนนจำลองกลุ่มความเสี่ยงปานกลาง (เพื่อการสาธิต)"),
    (300, "ต้องปรับปรุง (Poor)", "ระดับคะแนนจำลองกลุ่มความเสี่ยงสูง (เพื่อการสาธิต)"),
]


def credit_band(score: float):
    """คืนค่า (ชื่อระดับ, คำอธิบาย) ของคะแนนเครดิตที่กำหนด"""
    for min_score, name, desc in CREDIT_BANDS:
        if score >= min_score:
            return name, desc
    return CREDIT_BANDS[-1][1], CREDIT_BANDS[-1][2]


FEATURE_SCHEMA = [
    {"key": "monthly_income", "label": "รายได้ต่อเดือน [685–7408]", "type": "number", "min": 0.0, "max": 22224.0, "default": 4008.0, "step": 50.0, "core": True, "training_min": 685.0, "training_max": 7408.0},
    {"key": "monthly_expense_total", "label": "รายจ่ายรวมต่อเดือน [159–5854]", "type": "number", "min": 0.0, "max": 17562.0, "default": 3024.0, "step": 50.0, "core": True, "training_min": 159.0, "training_max": 5854.0},
    {"key": "savings_rate", "label": "อัตราการออม (สัดส่วน) [0.05–0.4]", "type": "number", "min": 0.0, "max": 1.0, "default": 0.23, "step": 0.01, "core": True, "training_min": 0.05, "training_max": 0.4},
    {"key": "budget_goal", "label": "เป้าหมายงบประมาณต่อเดือน [1175–4387]", "type": "number", "min": 0.0, "max": 13161.0, "default": 2822.0, "step": 50.0, "core": False, "training_min": 1175.0, "training_max": 4387.0},
    {"key": "financial_scenario", "label": "สถานการณ์ทางการเงิน", "type": "select", "options": ["normal", "inflation", "recession"], "format_map": {"normal": "ปกติ (Normal)", "inflation": "เงินเฟ้อ (Inflation)", "recession": "เศรษฐกิจถดถอย (Recession)"}, "core": False},
    {"key": "debt_to_income_ratio", "label": "สัดส่วนหนี้สินต่อรายได้ [0.1–0.6]", "type": "number", "min": 0.0, "max": 1.0, "default": 0.35, "step": 0.01, "core": True, "training_min": 0.1, "training_max": 0.6},
    {"key": "loan_payment", "label": "ยอดผ่อนชำระหนี้ต่อเดือน [0–1177]", "type": "number", "min": 0.0, "max": 3531.0, "default": 509.0, "step": 10.0, "core": True, "training_min": 0.0, "training_max": 1177.0},
    {"key": "investment_amount", "label": "ยอดเงินลงทุนต่อเดือน [0–1293]", "type": "number", "min": 0.0, "max": 3879.0, "default": 388.0, "step": 10.0, "core": False, "training_min": 0.0, "training_max": 1293.0},
    {"key": "subscription_services", "label": "จำนวนบริการสมาชิก (รายการ) [1–9]", "type": "number", "min": 0.0, "max": 27.0, "default": 5.0, "step": 1.0, "core": False, "training_min": 1.0, "training_max": 9.0},
    {"key": "emergency_fund", "label": "เงินสำรองฉุกเฉิน [0–2586]", "type": "number", "min": 0.0, "max": 7758.0, "default": 1010.0, "step": 50.0, "core": True, "training_min": 0.0, "training_max": 2586.0},
    {"key": "transaction_count", "label": "จำนวนธุรกรรมต่อเดือน (ครั้ง) [20–99]", "type": "number", "min": 0.0, "max": 297.0, "default": 60.0, "step": 1.0, "core": False, "training_min": 20.0, "training_max": 99.0},
    {"key": "fraud_flag", "label": "พบธุรกรรมผิดปกติ/ฉ้อโกงหรือไม่", "type": "select", "options": ["0", "1"], "format_map": {"0": "ไม่พบ (0)", "1": "พบ (1)"}, "core": False},
    {"key": "discretionary_spending", "label": "รายจ่ายฟุ่มเฟือย/ไม่จำเป็นต่อเดือน [0–1159]", "type": "number", "min": 0.0, "max": 3477.0, "default": 496.0, "step": 10.0, "core": False, "training_min": 0.0, "training_max": 1159.0},
    {"key": "essential_spending", "label": "รายจ่ายจำเป็นต่อเดือน [1000–4246]", "type": "number", "min": 0.0, "max": 12738.0, "default": 2216.0, "step": 50.0, "core": False, "training_min": 1000.0, "training_max": 4246.0},
    {"key": "income_type", "label": "ประเภทรายได้", "type": "select", "options": ["Salary", "Freelance", "Mixed"], "format_map": {"Salary": "เงินเดือนประจำ (Salary)", "Freelance": "ฟรีแลนซ์ (Freelance)", "Mixed": "รายได้ผสม (Mixed)"}, "core": False},
    {"key": "rent_or_mortgage", "label": "ค่าเช่า/ผ่อนบ้านต่อเดือน [300–2615]", "type": "number", "min": 0.0, "max": 7845.0, "default": 1214.0, "step": 10.0, "core": False, "training_min": 300.0, "training_max": 2615.0},
    {"key": "category", "label": "หมวดรายจ่ายหลัก", "type": "select", "options": ["Dining Out", "Education", "Entertainment", "Groceries", "Healthcare", "Insurance", "Investments", "Rent", "Transportation", "Utilities"], "format_map": {"Dining Out": "กินข้าวนอกบ้าน (Dining Out)", "Education": "การศึกษา (Education)", "Entertainment": "บันเทิง (Entertainment)", "Groceries": "ของใช้/ของกินในบ้าน (Groceries)", "Healthcare": "สุขภาพ (Healthcare)", "Insurance": "ประกัน (Insurance)", "Investments": "การลงทุน (Investments)", "Rent": "ค่าเช่า (Rent)", "Transportation": "เดินทาง (Transportation)", "Utilities": "ค่าน้ำค่าไฟ (Utilities)"}, "core": False},
    {"key": "cash_flow_status", "label": "สถานะกระแสเงินสด", "type": "select", "options": ["Positive", "Neutral", "Negative"], "format_map": {"Positive": "เป็นบวก (Positive)", "Neutral": "เท่าทุน (Neutral)", "Negative": "ติดลบ (Negative)"}, "core": True},
    {"key": "financial_advice_score", "label": "คะแนนคำแนะนำทางการเงิน (0-100)", "type": "number", "min": 0.0, "max": 300.0, "default": 49.0, "step": 1.0, "core": False, "training_min": 0.0, "training_max": 100.0},
    {"key": "actual_savings", "label": "เงินออมจริงสะสม [0–6590]", "type": "number", "min": 0.0, "max": 19770.0, "default": 1033.0, "step": 50.0, "core": False, "training_min": 0.0, "training_max": 6590.0},
    {"key": "savings_goal_met", "label": "บรรลุเป้าหมายการออมหรือไม่", "type": "select", "options": ["0", "1"], "format_map": {"0": "ยังไม่บรรลุ (0)", "1": "บรรลุแล้ว (1)"}, "core": False},
    {"key": "financial_stress_level", "label": "ระดับความเครียดทางการเงิน", "type": "select", "options": ["Low", "Medium", "High"], "format_map": {"Low": "ต่ำ (Low)", "Medium": "ปานกลาง (Medium)", "High": "สูง (High)"}, "core": False},
]


# ------------------------------------------------------------------------------
# 2) ฟังก์ชันช่วยเหลือ: โหลดโมเดล / เข้ารหัสฟีเจอร์
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner="กำลังโหลดโมเดล...")
def load_model(path: str):
    return joblib.load(path)


def build_input_table(model, raw_values: dict) -> Table:
    raw_vars = []
    for attr in model.original_domain.attributes:
        var = attr
        while var.compute_value is not None and hasattr(var.compute_value, "variable"):
            var = var.compute_value.variable
        raw_vars.append(var)

    missing = [v.name for v in raw_vars if v.name not in raw_values]
    if missing:
        raise ValueError(
            "โมเดลต้องใช้ฟีเจอร์ที่ฟอร์มไม่มีช่องกรอก: " + ", ".join(missing)
        )

    row = []
    for v in raw_vars:
        value = raw_values[v.name]
        if v.is_discrete:
            row.append(float(v.values.index(str(value))))
        else:
            row.append(float(value))
    return Table.from_numpy(Domain(raw_vars), np.array([row], dtype=float))


# ------------------------------------------------------------------------------
# 3) ส่วนหน้าเว็บแอป (UI)
# ------------------------------------------------------------------------------
st.set_page_config(page_title="AI วิเคราะห์ความสามารถบริหารรายรับรายจ่าย", page_icon="💰")

# หัวข้อหลักของแอป
st.title("AI วิเคราะห์ความสามารถบริหารรายรับรายจ่าย")

# 🚨 [เพิ่มจุดที่ 1] แสดงกล่องคำเตือนถาวรเรื่องความแม่นยำและ R2 ตามฟีดแบ็กอาจารย์ (ข้อ 3.15)
st.warning(
    """
    **คำเตือนเรื่องความแม่นยำและการใช้งาน (ข้อความเตือนถาวร):**
    
    ตัวแบบนี้ฝึกจากข้อมูลสังเคราะห์และมีค่า **$R^2 = -0.018$** ซึ่ง**ต่ำกว่าการทำนายด้วยค่าเฉลี่ย** 
    ผลทำนายที่แสดงจึงใกล้เคียงค่าเฉลี่ยสำหรับผู้ใช้เกือบทุกคน (ประมาณ 680 คะแนน) 
    
    ระบบนี้จัดทำขึ้นเพื่อ**สาธิตการนำตัวแบบจาก Orange Data Mining มาใช้งานบนเว็บเท่านั้น** 
    **ไม่สามารถนำไปใช้ประเมิน หรือตัดสินการขออนุมัติสินเชื่อจริงได้**
    """
)

st.caption(
    "กรอกข้อมูลทางการเงินของคุณ แล้วให้โมเดล Linear Regression ที่ฝึกไว้ทำนาย "
    "คะแนนเครดิต (credit_score) ในช่วงมาตรฐาน 300-850"
)

# --- 3.1 เลือกโมเดล ---
st.subheader("1) เลือกโมเดลที่ต้องการใช้ทำนาย")

MODEL_CHOICES = {
    "linear.pkcls": "Linear Regression",
    "Linear.pkcls": "Linear Regression",
}

files_on_disk = {f.lower(): f for f in os.listdir(MODEL_DIR)}
available_models = {
    files_on_disk[fname.lower()]: label
    for fname, label in MODEL_CHOICES.items()
    if fname.lower() in files_on_disk
}

model_path = None
if available_models:
    model_choice = st.selectbox(
        "เลือกโมเดล (.pkcls)",
        list(available_models.keys()),
        format_func=lambda fname: f"{available_models[fname]} ({fname})",
    )
    model_path = os.path.join(MODEL_DIR, model_choice)
else:
    st.warning(
        "ไม่พบไฟล์โมเดล linear.pkcls — กรุณาวางไฟล์ไว้ในโฟลเดอร์เดียวกับ app.py"
    )

model = None
if model_path:
    try:
        model = load_model(model_path)
        st.success(f"โหลดโมเดลสำเร็จ: {os.path.basename(model_path)}")
    except Exception as e:
        st.error(f"โหลดโมเดลไม่สำเร็จ: {e}")

# --- 3.2 ฟอร์มกรอกข้อมูล ---
st.subheader("2) กรอกข้อมูลทางการเงินของคุณ")
st.caption(
    "ตัวเลขในวงเล็บเหลี่ยม [ต่ำสุด–สูงสุด] คือช่วงที่พบในข้อมูลที่ใช้ฝึกโมเดล"
)


def render_field(feat, container):
    with container:
        if feat["type"] == "number":
            return st.number_input(
                feat["label"],
                min_value=feat["min"],
                max_value=feat["max"],
                value=feat["default"],
                step=feat["step"],
                key=feat["key"],
            )
        else:
            fmt = feat.get("format_map")
            return st.selectbox(
                feat["label"],
                feat["options"],
                format_func=(lambda v, fmt=fmt: fmt.get(v, v) if fmt else v),
                key=feat["key"],
            )


core_features = [f for f in FEATURE_SCHEMA if f.get("core")]
extra_features = [f for f in FEATURE_SCHEMA if not f.get("core")]

user_inputs = {}

col1, col2 = st.columns(2)
for i, feat in enumerate(core_features):
    target_col = col1 if i % 2 == 0 else col2
    user_inputs[feat["key"]] = render_field(feat, target_col)

with st.expander("ข้อมูลเพิ่มเติม (ไม่บังคับปรับ — ถ้าไม่แก้ไขจะใช้ค่าเริ่มต้น)"):
    ecol1, ecol2 = st.columns(2)
    for i, feat in enumerate(extra_features):
        target_col = ecol1 if i % 2 == 0 else ecol2
        user_inputs[feat["key"]] = render_field(feat, target_col)

# --- 3.3 ปุ่มทำนายผล ---
st.subheader("3) ทำนายผล")
predict_clicked = st.button("ทำนายผล", type="primary", use_container_width=True)

if predict_clicked:
    if model is None:
        st.error("กรุณาเลือกโมเดลก่อนกดทำนายผล")
    else:
        try:
            out_of_range = []
            for feat in FEATURE_SCHEMA:
                if feat["type"] != "number" or "training_min" not in feat:
                    continue
                v = user_inputs[feat["key"]]
                lo, hi = feat["training_min"], feat["training_max"]
                if v < lo or v > hi:
                    out_of_range.append(f"{feat['label'].split(' [')[0]} = {v:g} (ช่วงฝึก {lo:g}–{hi:g})")
            if out_of_range:
                st.warning(
                    "ค่าต่อไปนี้อยู่นอกช่วงที่โมเดลเคยเห็นตอนฝึก:\n\n- "
                    + "\n- ".join(out_of_range)
                )

            table = build_input_table(model, user_inputs)
            raw_score = float(model(table)[0])

            score = min(max(raw_score, SCORE_MIN), SCORE_MAX)
            band_name, band_desc = credit_band(score)

            # 🚨 [เพิ่มจุดที่ 2] แสดงผลทำนายพร้อมคำเตือนกำกับชัดเจน
            st.success(f"คะแนนเครดิตที่ทำนายได้ (ค่าสาธิต): **{score:.0f}** คะแนน — ระดับ {band_name}")
            st.info(f"**คำอธิบายระดับ:** {band_desc}")

            st.progress((score - SCORE_MIN) / (SCORE_MAX - SCORE_MIN))
            st.caption(f"ช่วงคะแนนมาตรฐาน {SCORE_MIN} - {SCORE_MAX}")

            # 🚨 [เพิ่มจุดที่ 3] ย้ำคำเตือนความแม่นยำทุกครั้งเมื่อกดทำนายผลสำเร็จ
            st.warning(
                "⚠️ **หมายเหตุซ้ำ:** ตัวแบบมี $R^2 = -0.018$ ผลลัพธ์นี้ใกล้เคียงค่าเฉลี่ย "
                "และใช้เพื่อการสาธิตระบบเท่านั้น ไม่สามารถใช้อ้างอิงทางกฎหมายหรือการขอสินเชื่อได้"
            )

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดระหว่างการทำนาย: {e}")

st.divider()
st.caption(
    "ระบบต้นแบบวิจัย - พัฒนาโดย Orange Data Mining & Streamlit"
)
