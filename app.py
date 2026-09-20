"""
================================================================================
 AI วิเคราะห์ความสามารถบริหารรายรับรายจ่าย
--------------------------------------------------------------------------------
 แอปนี้โหลดโมเดลที่ฝึกไว้จาก Orange Data Mining (ไฟล์ .pkcls) แล้วให้ผู้ใช้
 กรอกข้อมูลทางการเงินของตนเอง เพื่อทำนาย "ระดับความเครียดทางการเงิน"
 (financial_stress_level: High / Medium / Low)

 หมายเหตุสำคัญ (โปรดอ่าน):
 ไฟล์ .pkcls ที่แนบมา (Logistic.pkcls, Neural.pkcls, Tree.pkcls) ไม่ใช่ไฟล์
 scikit-learn ธรรมดา แต่เป็น "โมเดลของ Orange3" (Orange.classification.*)
 ที่ถูก pickle ออกมาจากวิดเจ็ต Save Model ในไฟล์เวิร์กโฟลว์ AI.ows
 ดังนั้นแม้จะใช้คำสั่ง joblib.load() ตามที่ขอ แต่เบื้องหลังต้องพึ่งพา
 ไลบรารี Orange3 (แพ็กเกจ PyPI ชื่อ "Orange3") ในการ "แกะ" คลาสของโมเดลออกมา
 มิฉะนั้นจะได้ข้อผิดพลาด "No module named 'Orange'"
 => จึงต้องติดตั้ง Orange3 เพิ่มเติม (ระบุไว้ใน requirements.txt แล้ว)

 ข้อดีคือ Orange เก็บ "โครงสร้างข้อมูล (Domain)" ของฟีเจอร์ที่ใช้ตอนฝึกโมเดล
 ไว้ในตัวโมเดลเอง (model.domain.attributes) ทำให้แอปนี้สามารถ "อ่านชื่อ/ชนิด/
 การเข้ารหัส one-hot" ของแต่ละโมเดลได้อัตโนมัติ โดยไม่ต้องเขียน mapping
 แยกสำหรับแต่ละโมเดล (โค้ดส่วน encode_features() ด้านล่างทำหน้าที่นี้)
================================================================================
"""

import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------------------
# 1) กำหนดค่าคงที่ / โครงสร้างฟีเจอร์ (schema) ตามชุดข้อมูลที่ใช้ฝึกโมเดล
# ------------------------------------------------------------------------------
# โฟลเดอร์ที่เก็บไฟล์โมเดล .pkcls ทั้ง 3 ไฟล์ (Logistic.pkcls, Neural.pkcls,
# Tree.pkcls) ให้ดาวน์โหลดจาก Google Drive โฟลเดอร์ Model มาวางไว้ในโฟลเดอร์
# เดียวกับ app.py นี้ (หรือเปลี่ยนพาธด้านล่างถ้าจะเก็บไว้ในโฟลเดอร์ย่อย)
MODEL_DIR = "."

# ชื่อคอลัมน์ผลลัพธ์ (target) และค่าที่เป็นไปได้ -- อ้างอิงจากตัวโมเดลจริง
# (financial_stress_level มีค่า: High, Low, Medium)
TARGET_NAME = "financial_stress_level"

# คำอธิบาย/ป้ายกำกับภาษาไทยของแต่ละคลาสผลลัพธ์ (ใช้ตอนแสดงผล ปรับแก้ได้)
TARGET_LABEL_TH = {
    "High": "สูง (มีความเสี่ยงด้านความเครียดทางการเงินสูง)",
    "Medium": "ปานกลาง",
    "Low": "ต่ำ (บริหารการเงินอยู่ในเกณฑ์ดี)",
}

# โครงสร้างฟีเจอร์ต้น (features) ทั้งหมด "ในรูปแบบดิบ" ก่อนเข้ารหัส
# ตรงกับคอลัมน์ต้นฉบับที่ใช้ตอนฝึกโมเดลใน Orange (ดูจากไฟล์ AI.ows / โมเดล Tree)
# type:
#   "number"  -> ใช้ st.number_input (ตัวแปรต่อเนื่อง / ตัวเลข)
#   "select"  -> ใช้ st.selectbox   (ตัวแปรหมวดหมู่ / ข้อความ)
FEATURE_SCHEMA = [
    {"key": "monthly_income", "label": "รายได้ต่อเดือน (บาท)", "type": "number",
     "min": 0.0, "max": 1_000_000.0, "default": 30000.0, "step": 500.0},
    {"key": "monthly_expense_total", "label": "รายจ่ายรวมต่อเดือน (บาท)", "type": "number",
     "min": 0.0, "max": 1_000_000.0, "default": 20000.0, "step": 500.0},
    {"key": "savings_rate", "label": "อัตราการออม (สัดส่วน 0-1)", "type": "number",
     "min": 0.0, "max": 1.0, "default": 0.15, "step": 0.01},
    {"key": "budget_goal", "label": "เป้าหมายงบประมาณต่อเดือน (บาท)", "type": "number",
     "min": 0.0, "max": 1_000_000.0, "default": 5000.0, "step": 500.0},
    {"key": "financial_scenario", "label": "สถานการณ์ทางการเงิน", "type": "select",
     "options": ["normal", "inflation", "recession"]},
    {"key": "credit_score", "label": "คะแนนเครดิต (Credit Score)", "type": "number",
     "min": 300.0, "max": 850.0, "default": 650.0, "step": 1.0},
    {"key": "debt_to_income_ratio", "label": "สัดส่วนหนี้สินต่อรายได้ (0-1)", "type": "number",
     "min": 0.0, "max": 2.0, "default": 0.3, "step": 0.01},
    {"key": "loan_payment", "label": "ยอดผ่อนชำระหนี้ต่อเดือน (บาท)", "type": "number",
     "min": 0.0, "max": 1_000_000.0, "default": 3000.0, "step": 500.0},
    {"key": "investment_amount", "label": "ยอดเงินลงทุนต่อเดือน (บาท)", "type": "number",
     "min": 0.0, "max": 1_000_000.0, "default": 2000.0, "step": 500.0},
    {"key": "subscription_services", "label": "ค่าบริการสมาชิกรายเดือน (บาท)", "type": "number",
     "min": 0.0, "max": 100_000.0, "default": 500.0, "step": 100.0},
    {"key": "emergency_fund", "label": "เงินสำรองฉุกเฉิน (บาท)", "type": "number",
     "min": 0.0, "max": 5_000_000.0, "default": 10000.0, "step": 500.0},
    {"key": "transaction_count", "label": "จำนวนธุรกรรมต่อเดือน", "type": "number",
     "min": 0.0, "max": 1000.0, "default": 40.0, "step": 1.0},
    {"key": "fraud_flag", "label": "พบธุรกรรมผิดปกติ/ฉ้อโกงหรือไม่", "type": "select",
     "options": ["0", "1"], "format_map": {"0": "ไม่พบ (0)", "1": "พบ (1)"}},
    {"key": "discretionary_spending", "label": "รายจ่ายฟุ่มเฟือย/ไม่จำเป็นต่อเดือน (บาท)", "type": "number",
     "min": 0.0, "max": 1_000_000.0, "default": 4000.0, "step": 500.0},
    {"key": "essential_spending", "label": "รายจ่ายจำเป็นต่อเดือน (บาท)", "type": "number",
     "min": 0.0, "max": 1_000_000.0, "default": 15000.0, "step": 500.0},
    {"key": "income_type", "label": "ประเภทรายได้", "type": "select",
     "options": ["Salary", "Freelance", "Mixed"]},
    {"key": "rent_or_mortgage", "label": "ค่าเช่า/ผ่อนบ้านต่อเดือน (บาท)", "type": "number",
     "min": 0.0, "max": 1_000_000.0, "default": 8000.0, "step": 500.0},
    {"key": "category", "label": "หมวดรายจ่ายหลัก", "type": "select",
     "options": ["Dining Out", "Education", "Entertainment", "Groceries", "Healthcare",
                 "Insurance", "Investments", "Rent", "Transportation", "Utilities"]},
    {"key": "cash_flow_status", "label": "สถานะกระแสเงินสด", "type": "select",
     "options": ["Positive", "Neutral", "Negative"]},
    {"key": "financial_advice_score", "label": "คะแนนคำแนะนำทางการเงิน", "type": "number",
     "min": 0.0, "max": 100.0, "default": 60.0, "step": 1.0},
    {"key": "actual_savings", "label": "เงินออมจริงสะสม (บาท)", "type": "number",
     "min": 0.0, "max": 10_000_000.0, "default": 15000.0, "step": 500.0},
    {"key": "savings_goal_met", "label": "บรรลุเป้าหมายการออมหรือไม่", "type": "select",
     "options": ["0", "1"], "format_map": {"0": "ยังไม่บรรลุ (0)", "1": "บรรลุแล้ว (1)"}},
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


def encode_features(model, raw_values: dict) -> np.ndarray:
    """
    แปลงค่าที่ผู้ใช้กรอก (raw_values: dict ตาม FEATURE_SCHEMA)
    ให้อยู่ในรูปแบบเวกเตอร์ตัวเลขที่ "ตรงกับตอนฝึกโมเดล" แบบอัตโนมัติ
    โดยอ่านโครงสร้าง domain จากตัวโมเดลเอง (model.domain.attributes)

    กติกาการแปลง (สอดคล้องกับวิธี Orange เตรียมข้อมูลก่อนฝึกโมเดล):
      - ถ้าเป็นตัวแปรต่อเนื่อง (ContinuousVariable) และชื่อคอลัมน์ไม่มีเครื่องหมาย '='
        -> ใช้ค่าตัวเลขที่ผู้ใช้กรอกตรง ๆ
      - ถ้าเป็นตัวแปรที่ถูก one-hot encode ไว้ล่วงหน้า (ชื่อรูปแบบ "คอลัมน์=ค่า"
        เช่น "category=Groceries") -> ใส่ 1.0 ถ้าค่าที่ผู้ใช้เลือกตรงกับ "ค่า"
        นั้น มิฉะนั้นใส่ 0.0 (นี่คือ one-hot encoding แบบเดียวกับตอนฝึกโมเดล
        Logistic Regression / Neural Network ใน Orange)
      - ถ้าเป็นตัวแปรหมวดหมู่แบบไม่ได้ถูกแยกคอลัมน์ (DiscreteVariable ปกติ เช่น
        ในโมเดล Tree ที่เก็บ category เป็นคอลัมน์เดียว) -> แปลงเป็นรหัสตัวเลข
        ตามลำดับ index ใน attr.values (เหมือนวิธีที่ Orange เก็บข้อมูลภายใน)
    """
    row = []
    for attr in model.domain.attributes:
        name = attr.name
        if attr.is_discrete:
            # กรณีตัวแปรหมวดหมู่แบบคอลัมน์เดียว (ไม่ถูกแยก one-hot ไว้ล่วงหน้า)
            base_key = name
            user_val = str(raw_values.get(base_key, attr.values[0]))
            code = attr.values.index(user_val) if user_val in attr.values else 0
            row.append(float(code))
        elif "=" in name:
            # กรณีตัวแปรต่อเนื่องที่จริง ๆ แล้วคือผลลัพธ์ของการ one-hot encode
            # เช่น "income_type=Salary" -> ต้อง match กับคอลัมน์ต้นฉบับ "income_type"
            base_key, category_value = name.split("=", 1)
            user_val = str(raw_values.get(base_key, ""))
            row.append(1.0 if user_val == category_value else 0.0)
        else:
            # ตัวแปรตัวเลขต่อเนื่องปกติ
            row.append(float(raw_values.get(name, 0.0)))
    return np.array([row], dtype=float)


# ------------------------------------------------------------------------------
# 3) ส่วนหน้าเว็บแอป (UI)
# ------------------------------------------------------------------------------
st.set_page_config(page_title="AI วิเคราะห์ความสามารถบริหารรายรับรายจ่าย", page_icon="💰")

# หัวข้อหลักของแอปตามที่กำหนด
st.title("AI วิเคราะห์ความสามารถบริหารรายรับรายจ่าย")
st.caption(
    "กรอกข้อมูลทางการเงินของคุณ แล้วให้โมเดล AI ที่ฝึกไว้ทำนายระดับความเครียด "
    "ทางการเงิน (financial_stress_level)"
)

# --- 3.1 เลือกโมเดล ---
st.subheader("1) เลือกโมเดลที่ต้องการใช้ทำนาย")

# รายชื่อโมเดลที่มีมาให้ตายตัว 3 ไฟล์ (วางไฟล์ .pkcls ทั้ง 3 นี้ไว้ในโฟลเดอร์
# เดียวกับ app.py ก่อนรันแอป ไม่ต้องอัปโหลดผ่านหน้าเว็บ)
# key = ชื่อไฟล์จริง, value = ชื่อที่แสดงในหน้าจอ (แก้ข้อความให้สวยงามได้ตามใจ)
MODEL_CHOICES = {
    "Logistic.pkcls": "Logistic Regression",
    "Neural.pkcls": "Neural Network",
    "Tree.pkcls": "Decision Tree",
}

# เลือกเฉพาะไฟล์ที่มีอยู่จริงในโฟลเดอร์ของแอปเท่านั้น (กันแอปพังถ้าไฟล์ยังไม่ครบ)
available_models = {
    fname: label
    for fname, label in MODEL_CHOICES.items()
    if os.path.exists(os.path.join(MODEL_DIR, fname))
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
        "ไม่พบไฟล์โมเดล Logistic.pkcls / Neural.pkcls / Tree.pkcls "
        "กรุณาวางไฟล์ทั้ง 3 (ดาวน์โหลดจากโฟลเดอร์ Model บน Google Drive) "
        "ไว้ในโฟลเดอร์เดียวกับ app.py แล้วรันแอปใหม่อีกครั้ง"
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

user_inputs = {}
# แบ่งเป็น 2 คอลัมน์ให้ฟอร์มดูกระชับขึ้น
col1, col2 = st.columns(2)
for i, feat in enumerate(FEATURE_SCHEMA):
    target_col = col1 if i % 2 == 0 else col2
    with target_col:
        if feat["type"] == "number":
            user_inputs[feat["key"]] = st.number_input(
                feat["label"],
                min_value=feat["min"],
                max_value=feat["max"],
                value=feat["default"],
                step=feat["step"],
                key=feat["key"],
            )
        else:  # select
            fmt = feat.get("format_map")
            user_inputs[feat["key"]] = st.selectbox(
                feat["label"],
                feat["options"],
                format_func=(lambda v, fmt=fmt: fmt.get(v, v) if fmt else v),
                key=feat["key"],
            )

# --- 3.3 ปุ่มทำนายผล ---
st.subheader("3) ทำนายผล")
predict_clicked = st.button("ทำนายผล", type="primary", use_container_width=True)

if predict_clicked:
    if model is None:
        st.error("กรุณาเลือก/อัปโหลดโมเดลก่อนกดทำนายผล")
    else:
        try:
            # แปลงค่าที่กรอกให้ตรงรูปแบบตอนฝึกโมเดล (one-hot ตามความจำเป็น)
            X = encode_features(model, user_inputs)

            # ทำนายคลาสผลลัพธ์
            pred_idx = int(model(X)[0])
            pred_label = model.domain.class_var.values[pred_idx]

            # ทำนายความน่าจะเป็นของแต่ละคลาส (ถ้าโมเดลรองรับ)
            try:
                probs = model(X, model.Probs)[0]
            except Exception:
                probs = None

            st.success(
                f"ผลการทำนาย: ระดับความเครียดทางการเงิน = **{pred_label}** "
                f"({TARGET_LABEL_TH.get(pred_label, pred_label)})"
            )

            if probs is not None:
                st.write("ความน่าจะเป็นของแต่ละระดับ:")
                prob_df = pd.DataFrame(
                    {
                        "ระดับ": model.domain.class_var.values,
                        "ความน่าจะเป็น": probs,
                    }
                ).sort_values("ความน่าจะเป็น", ascending=False)
                st.dataframe(prob_df, hide_index=True, use_container_width=True)
                st.bar_chart(prob_df.set_index("ระดับ"))
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดระหว่างการทำนาย: {e}")

st.divider()
st.caption(
    "หมายเหตุ: โมเดลนี้ฝึกด้วย Orange Data Mining แล้วบันทึกเป็น .pkcls "
    "แอปนี้จึงต้องติดตั้งไลบรารี Orange3 เพิ่มเติมนอกเหนือจาก scikit-learn "
    "เพื่อให้ joblib โหลดโมเดลได้ถูกต้อง"
)
