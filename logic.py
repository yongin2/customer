import pickle
import random
import warnings
import os

warnings.filterwarnings("ignore")

import numpy as np
import Orange
from Orange.data import Domain, ContinuousVariable, DiscreteVariable, Table

# ---------------------------------------------------------------------------
# 1. 모델 로드
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

MODEL_PATHS = {
    "vip": os.path.join(MODEL_DIR, "vip_model_rf.pkcls"),
    "interest": os.path.join(MODEL_DIR, "interest_model_nb.pkcls"),
    "discount": os.path.join(MODEL_DIR, "discount_model_lr.pkcls"),
}


def load_models():
    models = {}
    for key, path in MODEL_PATHS.items():
        with open(path, "rb") as f:
            models[key] = pickle.load(f)
    return models


# ---------------------------------------------------------------------------
# 2. 입력 스키마 (오렌지 원본 데이터셋과 동일한 컬럼 구조)
# ---------------------------------------------------------------------------
EDU_VALUES = ("2n Cycle", "Basic", "Graduation", "Master", "PhD")
MARITAL_VALUES = ("Absurd", "Alone", "Divorced", "Married", "Single", "Together", "Widow", "YOLO")

EDU_LABELS_KO = [
    ("기초 교육", "Basic"),
    ("석사 준비과정", "2n Cycle"),
    ("학사", "Graduation"),
    ("석사", "Master"),
    ("박사", "PhD"),
]
MARITAL_LABELS_KO = [
    ("미혼", "Single"),
    ("기혼", "Married"),
    ("동거", "Together"),
    ("이혼", "Divorced"),
    ("사별", "Widow"),
    ("기타(이상값)", "Alone"),
]
EDU_KO2EN = dict(EDU_LABELS_KO)
MARITAL_KO2EN = dict(MARITAL_LABELS_KO)
EDU_EN2KO = {v: k for k, v in EDU_LABELS_KO}
MARITAL_EN2KO = {v: k for k, v in MARITAL_LABELS_KO}

RAW_ATTRS = [
    ContinuousVariable("Year_Birth"),
    DiscreteVariable("Education", values=EDU_VALUES),
    DiscreteVariable("Marital_Status", values=MARITAL_VALUES),
    ContinuousVariable("Income"),
    ContinuousVariable("Kidhome"),
    ContinuousVariable("Teenhome"),
    ContinuousVariable("Recency"),
    ContinuousVariable("NumDealsPurchases"),
    ContinuousVariable("NumWebPurchases"),
    ContinuousVariable("NumCatalogPurchases"),
    ContinuousVariable("NumStorePurchases"),
    ContinuousVariable("NumWebVisitsMonth"),
    DiscreteVariable("AcceptedCmp3", values=("0", "1")),
    DiscreteVariable("AcceptedCmp4", values=("0", "1")),
    DiscreteVariable("AcceptedCmp5", values=("0", "1")),
    DiscreteVariable("AcceptedCmp1", values=("0", "1")),
    DiscreteVariable("AcceptedCmp2", values=("0", "1")),
    DiscreteVariable("Complain", values=("0", "1")),
    DiscreteVariable("Response", values=("0", "1")),
    ContinuousVariable("Z_CostContact"),
    ContinuousVariable("Z_Revenue"),
]
RAW_DOMAIN = Domain(RAW_ATTRS)

DEFAULTS = {
    "Z_CostContact": 3,
    "Z_Revenue": 11,
    "AcceptedCmp1": "0", "AcceptedCmp2": "0", "AcceptedCmp3": "0",
    "AcceptedCmp4": "0", "AcceptedCmp5": "0",
    "Complain": "0", "Response": "0",
}

CHURN_THRESHOLD_DAYS = 60  # 재유치(휴면) 대상 기준


def build_table(customer: dict) -> Table:
    data = {**DEFAULTS, **customer}
    row = []
    for a in RAW_ATTRS:
        v = data[a.name]
        if isinstance(a, DiscreteVariable):
            row.append(a.values.index(str(v)))
        else:
            row.append(float(v))
    return Table.from_numpy(RAW_DOMAIN, np.array([row], dtype=float))


# ---------------------------------------------------------------------------
# 3. 예측
# ---------------------------------------------------------------------------
def predict_customer(models: dict, customer: dict) -> dict:
    table = build_table(customer)
    result = {}
    for key in ["vip", "interest", "discount"]:
        model = models[key]
        pred = model(table)
        probs = model(table, ret=Orange.classification.Model.ValueProbs)[1][0]
        cls_values = model.domain.class_var.values
        result[key] = {
            "prediction": cls_values[int(pred[0])],
            "probabilities": {c: round(float(p), 4) for c, p in zip(cls_values, probs)},
        }
    return result


# ---------------------------------------------------------------------------
# 4. 추천 / 우선순위 / 태그 로직
# ---------------------------------------------------------------------------
PRODUCT_MAP = {
    "Health/Wellness": "건강기능식품 / 유기농 제품 라인",
    "Luxury": "프리미엄 & 리미티드 에디션 컬렉션",
    "Meat/BBQ": "정육 세트 / 바비큐 용품",
    "Sweet/Snack": "디저트 & 스낵 박스",
    "Wine/Gourmet": "와인 & 고급 미식 상품",
}


def build_promotion(vip: str, discount: str) -> str:
    if discount == "High":
        if vip == "VIP":
            return "VIP 전용 할인 쿠폰 발송 추천 (이탈 방지 우선)"
        return "일반 프로모션 / 쿠폰 발송 추천 (구매 전환 유도)"
    else:
        if vip == "VIP":
            return "할인 없이 신제품·한정판 우선 안내"
        return "프로모션 불필요 (할인 민감도 낮음, 브랜드 콘텐츠 위주 소통)"


def compute_scores(customer: dict, prediction: dict) -> dict:
    web = float(customer.get("NumWebPurchases", 0))
    cat = float(customer.get("NumCatalogPurchases", 0))
    store = float(customer.get("NumStorePurchases", 0))
    recency = float(customer.get("Recency", 50))

    freq_score = min(100, (web + cat + store) * 3)
    recency_score = max(0, 100 - min(recency, 100))
    priority_score = round(0.5 * freq_score + 0.5 * recency_score, 1)

    is_vip = prediction["vip"]["prediction"] == "VIP"
    is_churn_risk = recency >= CHURN_THRESHOLD_DAYS

    tags = []
    if is_vip:
        tags.append("⭐VIP")
    if is_churn_risk:
        tags.append("🔻이탈위험")

    return {
        "priority_score": priority_score,
        "is_vip": is_vip,
        "is_churn_risk": is_churn_risk,
        "tags": ", ".join(tags) if tags else "-",
    }


def analyze_customer(models: dict, customer: dict) -> dict:
    prediction = predict_customer(models, customer)
    vip = prediction["vip"]["prediction"]
    interest = prediction["interest"]["prediction"]
    discount = prediction["discount"]["prediction"]
    scores = compute_scores(customer, prediction)

    return {
        "prediction": prediction,
        "recommended_product": PRODUCT_MAP.get(interest, "일반 상품 라인"),
        "promotion": build_promotion(vip, discount),
        **scores,
    }


# ---------------------------------------------------------------------------
# 5. 랜덤 고객 생성
# ---------------------------------------------------------------------------
def random_customer() -> dict:
    return {
        "Year_Birth": random.randint(1945, 2003),
        "Education": random.choice(EDU_VALUES),
        "Marital_Status": random.choice(["Single", "Married", "Together", "Divorced", "Widow"]),
        "Income": random.randint(15000, 150000),
        "Kidhome": random.randint(0, 2),
        "Teenhome": random.randint(0, 2),
        "Recency": random.randint(0, 99),
        "NumDealsPurchases": random.randint(0, 15),
        "NumWebPurchases": random.randint(0, 20),
        "NumCatalogPurchases": random.randint(0, 20),
        "NumStorePurchases": random.randint(0, 20),
        "NumWebVisitsMonth": random.randint(0, 15),
        "AcceptedCmp1": random.choices(["0", "1"], weights=[85, 15])[0],
        "AcceptedCmp2": random.choices(["0", "1"], weights=[90, 10])[0],
        "AcceptedCmp3": random.choices(["0", "1"], weights=[85, 15])[0],
        "AcceptedCmp4": random.choices(["0", "1"], weights=[85, 15])[0],
        "AcceptedCmp5": random.choices(["0", "1"], weights=[85, 15])[0],
        "Complain": random.choices(["0", "1"], weights=[95, 5])[0],
        "Response": random.choices(["0", "1"], weights=[85, 15])[0],
    }
