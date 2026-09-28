"""
================================================================================
 AI วิเคราะห์ความสามารถบริหารรายรับรายจ่าย
--------------------------------------------------------------------------------
 แอปนี้โหลดโมเดล Linear Regression ที่ฝึกไว้จาก Orange Data Mining (ไฟล์ .pkcls)
 แล้วให้ผู้ใช้กรอกข้อมูลทางการเงินของตนเอง เพื่อทำนาย "คะแนนเครดิต"
 (credit_score: ค่าตัวเลขต่อเนื่อง ช่วงมาตรฐาน 300-850 แบบ FICO/VantageScore)
 ** เป็นงานแบบ Regression (ทำนายตัวเลข) ไม่ใช่ Classification แล้ว **

 หมายเหตุสำคัญ (โปรดอ่าน):
 ไฟล์ .pkcls ไม่ใช่ไฟล์ scikit-learn ธรรมดา แต่เป็น "โมเดลของ Orange3"
 (Orange.regression.*) ที่ถูก pickle ออกมาจากวิดเจ็ต Save Model ใน Orange
 ดังนั้นแม้จะใช้คำสั่ง joblib.load() ตามที่ขอ แต่เบื้องหลังต้องพึ่งพา
 ไลบรารี Orange3 (แพ็กเกจ PyPI ชื่อ "Orange3") ในการ "แกะ" คลาสของโมเดลออกมา
 มิฉะนั้นจะได้ข้อผิดพลาด "No module named 'Orange'"
 => จึงต้องติดตั้ง Orange3 เพิ่มเติม (ระบุไว้ใน requirements.txt แล้ว)

 ข้อดีคือ Orange เก็บ "โครงสร้างข้อมูล (Domain)" ของฟีเจอร์ที่ใช้ตอนฝึกโมเดล
 ไว้ในตัวโมเดลเอง (model.domain.attributes) ทำให้แอปนี้สามารถ "อ่านชื่อ/ชนิด/
 การเข้ารหัส one-hot" ของแต่ละโมเดลได้อัตโนมัติ โดยไม่ต้องเขียน mapping
 แยกสำหรับแต่ละโมเดล (โค้ดส่วน build_input_table() ด้านล่างทำหน้าที่นี้)
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
# โฟลเดอร์ที่เก็บไฟล์โมเดล .pkcls (Linear.pkcls) วางไว้ในโฟลเดอร์
# เดียวกับ app.py นี้ (หรือเปลี่ยนพาธด้านล่างถ้าจะเก็บไว้ในโฟลเดอร์ย่อย)
MODEL_DIR = "."

# ชื่อคอลัมน์ผลลัพธ์ (target) -- ตอนนี้เป็นตัวเลขต่อเนื่อง (Regression)
# หมายเหตุ: credit_score เป็น "ตัวแปรตาม" จึงต้อง "ไม่" อยู่ในช่องกรอกข้อมูล (FEATURE_SCHEMA)
TARGET_NAME = "credit_score"

# ช่วงคะแนนเครดิตมาตรฐาน (FICO / VantageScore) ใช้ตัดค่าที่โมเดลทำนายให้อยู่ในช่วงนี้
# (Linear Regression ไม่รู้จักขอบเขต จึงอาจทำนายเกินช่วงได้เล็กน้อย)
SCORE_MIN, SCORE_MAX = 300, 850

# เกณฑ์แบ่งระดับคะแนน: (คะแนนต่ำสุดของระดับ, ชื่อระดับ, คำอธิบาย)
# เรียงจากสูงไปต่ำ เพื่อให้เช็คจากบนลงล่างแล้วเจอระดับแรกที่เข้าเงื่อนไข
CREDIT_BANDS = [
    (800, "ดีเยี่ยม (Exceptional)", "ความเสี่ยงต่ำมาก อนุมัติง่ายและได้ดอกเบี้ยต่ำสุด"),
    (740, "ดีมาก (Very Good)", "ประวัติทางการเงินดี ได้รับข้อเสนอที่ดี"),
    (670, "ดี (Good)", "เป็นเกณฑ์เฉลี่ยที่สถาบันการเงินยอมรับได้"),
    (580, "พอใช้ (Fair)", "กลุ่มเสี่ยงกึ่งซับไพรม์ อาจกู้ผ่านยากหรือโดนดอกเบี้ยแพง"),
    (300, "ต้องปรับปรุง (Poor)", "ขอสินเชื่อหรือทำบัตรเครดิตได้ยากมาก"),
]


def credit_band(score: float):
    """คืนค่า (ชื่อระดับ, คำอธิบาย) ของคะแนนเครดิตที่กำหนด"""
    for min_score, name, desc in CREDIT_BANDS:
        if score >= min_score:
            return name, desc
    return CREDIT_BANDS[-1][1], CREDIT_BANDS[-1][2]

# โครงสร้างฟีเจอร์ต้น (features) ทั้งหมด "ในรูปแบบดิบ" ก่อนเข้ารหัส
# ตรงกับคอลัมน์ต้นฉบับที่ใช้ตอนฝึกโมเดลใน Orange (ดูจากไฟล์ AI.ows / โมเดล Tree)
# type:
#   "number"  -> ใช้ st.number_input (ตัวแปรต่อเนื่อง / ตัวเลข)
#   "select"  -> ใช้ st.selectbox   (ตัวแปรหมวดหมู่ / ข้อความ)
# "core": True  -> ฟิลด์สำคัญ แสดงในหน้าฟอร์มหลักเลย
# "core": False -> ฟิลด์รอง ซ่อนไว้ใต้ "ข้อมูลเพิ่มเติม (ไม่บังคับ)" แบบพับเก็บได้
#                  (ยังส่งเข้าโมเดลเหมือนเดิมทุกตัว แค่ไม่โชว์เด่นในหน้าแรก
#                  เพื่อให้ฟอร์มดูกระชับ ปรับ True/False ตรงนี้ได้ตามต้องการ)
FEATURE_SCHEMA = [
    {"key": "monthly_income", "label": "รายได้ต่อเดือน [685–7408]", "type": "number", "min": 685.0, "max": 7408.0, "default": 4008.0, "step": 50.0, "core": True},
    {"key": "monthly_expense_total", "label": "รายจ่ายรวมต่อเดือน [159–5854]", "type": "number", "min": 159.0, "max": 5854.0, "default": 3024.0, "step": 50.0, "core": True},
    {"key": "savings_rate", "label": "อัตราการออม (สัดส่วน) [0.05–0.4]", "type": "number", "min": 0.05, "max": 0.4, "default": 0.23, "step": 0.01, "core": True},
    {"key": "budget_goal", "label": "เป้าหมายงบประมาณต่อเดือน [1175–4387]", "type": "number", "min": 1175.0, "max": 4387.0, "default": 2822.0, "step": 50.0, "core": False},
    {"key": "financial_scenario", "label": "สถานการณ์ทางการเงิน", "type": "select", "options": ["normal", "inflation", "recession"], "core": False},
    {"key": "debt_to_income_ratio", "label": "สัดส่วนหนี้สินต่อรายได้ [0.1–0.6]", "type": "number", "min": 0.1, "max": 0.6, "default": 0.35, "step": 0.01, "core": True},
    {"key": "loan_payment", "label": "ยอดผ่อนชำระหนี้ต่อเดือน [0–1177]", "type": "number", "min": 0.0, "max": 1177.0, "default": 509.0, "step": 10.0, "core": True},
    {"key": "investment_amount", "label": "ยอดเงินลงทุนต่อเดือน [0–1293]", "type": "number", "min": 0.0, "max": 1293.0, "default": 388.0, "step": 10.0, "core": False},
    {"key": "subscription_services", "label": "จำนวนบริการสมาชิก (รายการ) [1–9]", "type": "number", "min": 1.0, "max": 9.0, "default": 5.0, "step": 1.0, "core": False},
    {"key": "emergency_fund", "label": "เงินสำรองฉุกเฉิน [0–2586]", "type": "number", "min": 0.0, "max": 2586.0, "default": 1010.0, "step": 50.0, "core": True},
    {"key": "transaction_count", "label": "จำนวนธุรกรรมต่อเดือน (ครั้ง) [20–99]", "type": "number", "min": 20.0, "max": 99.0, "default": 60.0, "step": 1.0, "core": False},
    {"key": "fraud_flag", "label": "พบธุรกรรมผิดปกติ/ฉ้อโกงหรือไม่", "type": "select", "options": ["0", "1"], "format_map": {"0": "ไม่พบ (0)", "1": "พบ (1)"}, "core": False},
    {"key": "discretionary_spending", "label": "รายจ่ายฟุ่มเฟือย/ไม่จำเป็นต่อเดือน [0–1159]", "type": "number", "min": 0.0, "max": 1159.0, "default": 496.0, "step": 10.0, "core": False},
    {"key": "essential_spending", "label": "รายจ่ายจำเป็นต่อเดือน [1000–4246]", "type": "number", "min": 1000.0, "max": 4246.0, "default": 2216.0, "step": 50.0, "core": False},
    {"key": "income_type", "label": "ประเภทรายได้", "type": "select", "options": ["Salary", "Freelance", "Mixed"], "core": False},
    {"key": "rent_or_mortgage", "label": "ค่าเช่า/ผ่อนบ้านต่อเดือน [300–2615]", "type": "number", "min": 300.0, "max": 2615.0, "default": 1214.0, "step": 10.0, "core": False},
    {"key": "category", "label": "หมวดรายจ่ายหลัก", "type": "select", "options": ["Dining Out", "Education", "Entertainment", "Groceries", "Healthcare", "Insurance", "Investments", "Rent", "Transportation", "Utilities"], "core": False},
    {"key": "cash_flow_status", "label": "สถานะกระแสเงินสด", "type": "select", "options": ["Positive", "Neutral", "Negative"], "core": True},
    {"key": "financial_advice_score", "label": "คะแนนคำแนะนำทางการเงิน (0-100)", "type": "number", "min": 0.0, "max": 100.0, "default": 49.0, "step": 1.0, "core": False},
    {"key": "actual_savings", "label": "เงินออมจริงสะสม [0–6590]", "type": "number", "min": 0.0, "max": 6590.0, "default": 1033.0, "step": 50.0, "core": False},
    {"key": "savings_goal_met", "label": "บรรลุเป้าหมายการออมหรือไม่", "type": "select", "options": ["0", "1"], "format_map": {"0": "ยังไม่บรรลุ (0)", "1": "บรรลุแล้ว (1)"}, "core": False},
    {"key": "financial_stress_level", "label": "ระดับความเครียดทางการเงิน", "type": "select", "options": ["Low", "Medium", "High"], "format_map": {"Low": "ต่ำ (Low)", "Medium": "ปานกลาง (Medium)", "High": "สูง (High)"}, "core": False},
]


# ------------------------------------------------------------------------------
# 2) ฟังก์ชันช่วยเหลือ: ค้นหาไฟล์โมเดล / โหลดโมเดล / เข้ารหัสฟีเจอร์
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner="กำลังโหลดโมเดล...")
def load_model(path: str):
    """
    โหลดไฟล์โมเดล .pkcls ด้วย joblib
    (ไฟล์นี้เป็นอ็อบเจ็กต์โมเดลของ Orange3 ดังนั้นต้องติดตั้งไลบรารี Orange3
    ไว้ในสภาพแวดล้อมด้วย ไม่เช่นนั้น joblib.load จะหาโมดูล 'Orange' ไม่เจอ)
    """
    return joblib.load(path)


def build_input_table(model, raw_values: dict) -> Table:
    """
    แปลงค่าที่ผู้ใช้กรอก (raw_values) เป็นตาราง Orange ใน "รูปแบบดิบ" ของข้อมูลต้นทาง
    แล้วให้ Orange ทำขั้นตอนเตรียมข้อมูลเดิมซ้ำเอง (Normalize จากวิดเจ็ต Preprocess,
    one-hot ของตัวแปรหมวดหมู่ ฯลฯ) ตอนเรียก model(table)

    ทำไมไม่ใส่ค่าเข้า model.domain ตรง ๆ เหมือนเดิม:
    โมเดลที่ฝึกหลังผ่าน Preprocess > Normalize จะคาดหวังตัวเลขที่ "ถูกปรับสเกลแล้ว"
    ถ้าส่งค่าดิบ (เช่น รายได้ 30000) เข้าไปตรง ๆ ผลทำนายจะผิดเพี้ยนมาก
    ขั้นตอน: เดินย้อนตัวแปรใน model.original_domain ไปหา "ตัวแปรต้นทางดิบ"
    ผ่าน compute_value.variable -> สร้าง Table ด้วยตัวแปรดิบเหล่านั้น
    """
    raw_vars = []
    for attr in model.original_domain.attributes:
        var = attr
        while var.compute_value is not None and hasattr(var.compute_value, "variable"):
            var = var.compute_value.variable
        raw_vars.append(var)

    # ถ้าโมเดลต้องการฟีเจอร์ที่ฟอร์มไม่มีช่องกรอก ให้แจ้งชื่อชัด ๆ (ไม่เติม 0 เงียบ ๆ)
    missing = [v.name for v in raw_vars if v.name not in raw_values]
    if missing:
        raise ValueError(
            "โมเดลต้องใช้ฟีเจอร์ที่ฟอร์มไม่มีช่องกรอก: " + ", ".join(missing)
            + " — ให้ตั้งคอลัมน์เหล่านี้เป็น Ignore/Meta ใน Orange แล้วฝึกโมเดลใหม่ "
            "หรือเพิ่มเข้า FEATURE_SCHEMA ใน app.py"
        )

    row = []
    for v in raw_vars:
        value = raw_values[v.name]
        if v.is_discrete:
            # ตัวแปรหมวดหมู่: เก็บเป็นลำดับ index ของค่าใน v.values (แบบที่ Orange ใช้ภายใน)
            row.append(float(v.values.index(str(value))))
        else:
            row.append(float(value))
    return Table.from_numpy(Domain(raw_vars), np.array([row], dtype=float))


# ------------------------------------------------------------------------------
# 3) ส่วนหน้าเว็บแอป (UI)
# ------------------------------------------------------------------------------
st.set_page_config(page_title="AI วิเคราะห์ความสามารถบริหารรายรับรายจ่าย", page_icon="💰")

# หัวข้อหลักของแอปตามที่กำหนด
st.title("AI วิเคราะห์ความสามารถบริหารรายรับรายจ่าย")
st.caption(
    "กรอกข้อมูลทางการเงินของคุณ แล้วให้โมเดล Linear Regression ที่ฝึกไว้ทำนาย "
    "คะแนนเครดิต (credit_score) ในช่วงมาตรฐาน 300-850"
)

# --- 3.1 เลือกโมเดล ---
st.subheader("1) เลือกโมเดลที่ต้องการใช้ทำนาย")

# รายชื่อโมเดลที่มีมาให้ตายตัว (วางไฟล์ .pkcls ไว้ในโฟลเดอร์เดียวกับ app.py
# ก่อนรันแอป ไม่ต้องอัปโหลดผ่านหน้าเว็บ)
# key = ชื่อไฟล์จริงที่ Save Model จาก Orange (ต้องสะกดตรงตัวอักษรใหญ่-เล็ก)
# value = ชื่อที่แสดงในหน้าจอ
# อยากเพิ่มโมเดลอื่น (เช่น Ridge) ก็เพิ่มบรรทัดใหม่ตรงนี้ได้เลย
MODEL_CHOICES = {
    "Linear.pkcls": "Linear Regression",
}

# เลือกเฉพาะไฟล์ที่มีอยู่จริงในโฟลเดอร์ของแอปเท่านั้น (กันแอปพังถ้าไฟล์ยังไม่ครบ)
# หาไฟล์แบบไม่สนตัวพิมพ์เล็ก/ใหญ่ เพราะ Streamlit Cloud (Linux) แยก Linear.pkcls กับ
# linear.pkcls เป็นคนละไฟล์ แต่ชื่อที่ Orange Save ออกมาอาจเป็นตัวพิมพ์เล็ก
files_on_disk = {f.lower(): f for f in os.listdir(MODEL_DIR)}
available_models = {
    files_on_disk[fname.lower()]: label          # ใช้ชื่อไฟล์จริงบนดิสก์เป็น key
    for fname, label in MODEL_CHOICES.items()
    if fname.lower() in files_on_disk
}

# ถ้าไฟล์โมเดลบางไฟล์ (ไม่ใช่ทั้งหมด) หายไปจากโฟลเดอร์ ให้เตือนผู้ใช้ชัด ๆ
# ว่าไฟล์ไหนหายไปบ้าง จะได้ไม่งงว่าทำไม dropdown มีไม่ครบ 3 ตัวเลือก
missing_models = [f for f in MODEL_CHOICES if f.lower() not in files_on_disk]
if available_models and missing_models:
    st.info(
        "พบไฟล์โมเดลไม่ครบ ตอนนี้เจอเฉพาะ: "
        + ", ".join(available_models.keys())
        + f" — ยังหาไม่เจอ: {', '.join(missing_models)} "
        "กรุณาตรวจสอบว่าได้วางไฟล์นี้ไว้ในโฟลเดอร์เดียวกับ app.py "
        "(หรือ push ขึ้น repo ด้วย ถ้า deploy บน Streamlit Cloud) แล้วรีเฟรชแอป"
    )

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
        "ไม่พบไฟล์โมเดล linear.pkcls — กรุณาวางไฟล์ไว้ในโฟลเดอร์เดียวกับ app.py "
        "(ถ้า deploy บน Streamlit Cloud ต้อง push ไฟล์ขึ้น GitHub repo ด้วย) แล้วรีเฟรชแอป"
    )

model = None
if model_path:
    try:
        model = load_model(model_path)
        st.success(f"โหลดโมเดลสำเร็จ: {os.path.basename(model_path)} ({type(model).__name__})")
    except ModuleNotFoundError as e:
        st.error(
            f"โหลดโมเดลไม่สำเร็จ: ขาดไลบรารีที่จำเป็น ({e}). "
            "โมเดลนี้ถูกฝึกด้วย Orange3 กรุณาติดตั้ง Orange3 ตาม requirements.txt"
        )
    except Exception as e:
        st.error(f"โหลดโมเดลไม่สำเร็จ: {e}")

# --- 3.2 ฟอร์มกรอกข้อมูล ---
st.subheader("2) กรอกข้อมูลทางการเงินของคุณ")
st.caption("ช่วงค่าในวงเล็บเหลี่ยมอ้างอิงจากชุดข้อมูลที่ใช้ฝึกโมเดล — กรอกนอกช่วงนี้ผลทำนายจะไม่น่าเชื่อถือ")


def render_field(feat, container):
    """วาด widget 1 ช่อง (number_input หรือ selectbox) ตาม schema ของฟีเจอร์นั้น"""
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
        else:  # select
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

# ฟิลด์หลัก: แสดงตรง ๆ ในหน้าฟอร์ม (เฉพาะตัวที่สำคัญ/จำเป็นต่อการวิเคราะห์จริง ๆ)
col1, col2 = st.columns(2)
for i, feat in enumerate(core_features):
    target_col = col1 if i % 2 == 0 else col2
    user_inputs[feat["key"]] = render_field(feat, target_col)

# ฟิลด์รอง: พับเก็บไว้ใต้ expander เพื่อให้ฟอร์มหลักดูกระชับ
# (ถ้าไม่เปิดดู จะใช้ค่า default ที่ตั้งไว้ใน FEATURE_SCHEMA แทนโดยอัตโนมัติ)
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
            # แปลงค่าที่กรอกให้ตรงรูปแบบตอนฝึกโมเดล (one-hot ตามความจำเป็น)
            table = build_input_table(model, user_inputs)

            # Regression: model(X) คืน "ตัวเลขต่อเนื่อง" (ไม่มีคลาส/ความน่าจะเป็นแล้ว)
            raw_score = float(model(table)[0])

            # Linear Regression ไม่รู้จักขอบเขต จึงตัดค่าให้อยู่ในช่วงมาตรฐาน 300-850
            score = min(max(raw_score, SCORE_MIN), SCORE_MAX)
            band_name, band_desc = credit_band(score)

            st.success(f"คะแนนเครดิตที่ทำนายได้: **{score:.0f}** คะแนน — ระดับ {band_name}")
            st.write(band_desc)

            # แถบแสดงตำแหน่งของคะแนนในช่วง 300-850
            st.progress((score - SCORE_MIN) / (SCORE_MAX - SCORE_MIN))
            st.caption(f"ช่วงคะแนนมาตรฐาน {SCORE_MIN} - {SCORE_MAX}")

            # แจ้งกรณีค่าที่โมเดลคำนวณได้หลุดช่วง เพื่อความโปร่งใส
            if raw_score != score:
                st.caption(
                    f"(ค่าที่โมเดลคำนวณได้ดิบ ๆ คือ {raw_score:.1f} จึงปรับให้อยู่ในช่วงมาตรฐาน)"
                )
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดระหว่างการทำนาย: {e}")

st.divider()
st.caption(
    "หมายเหตุ: โมเดลนี้ฝึกด้วย Orange Data Mining แล้วบันทึกเป็น .pkcls "
    "แอปนี้จึงต้องติดตั้งไลบรารี Orange3 เพิ่มเติมนอกเหนือจาก scikit-learn "
    "เพื่อให้ joblib โหลดโมเดลได้ถูกต้อง"
)
