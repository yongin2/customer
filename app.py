import streamlit as st
import pandas as pd

import logic as L

st.set_page_config(page_title="고객 관리 AI", page_icon="🧑‍💼", layout="wide")


@st.cache_resource
def get_models():
    return L.load_models()


models = get_models()

st.title("🧑‍💼 고객 관리 AI 프로그램")
st.caption("VIP 여부 · 선호 카테고리 · 할인 민감성을 예측하고, 추천/우선순위/재유치 대상을 자동으로 판단합니다.")

tab1, tab2, tab3 = st.tabs(["고객 분석 (1명)", "여러 고객 비교", "재유치 대상 (이탈 위험)"])

# ---------------------------------------------------------------------------
# 탭 1. 고객 분석 (1명)
# ---------------------------------------------------------------------------
with tab1:
    DEFAULT_SINGLE = {
        "Year_Birth": 1985, "Education": "Graduation", "Marital_Status": "Married",
        "Income": 65000, "Kidhome": 0, "Teenhome": 1, "Recency": 10,
        "NumDealsPurchases": 3, "NumWebPurchases": 5, "NumCatalogPurchases": 4,
        "NumStorePurchases": 6, "NumWebVisitsMonth": 4, "Response": "1",
    }
    for k, v in DEFAULT_SINGLE.items():
        st.session_state.setdefault(f"s_{k}", v)

    def _apply_random_single():
        c = L.random_customer()
        for k in DEFAULT_SINGLE:
            st.session_state[f"s_{k}"] = c[k]

    col_input, col_result = st.columns([1, 1])

    with col_input:
        st.button("🎲 랜덤 고객 생성", on_click=_apply_random_single, key="btn_random_single")

        year_birth = st.number_input("출생연도", 1920, 2010, key="s_Year_Birth")
        edu_ko = st.selectbox("학력", [k for k, v in L.EDU_LABELS_KO],
                               index=[k for k, v in L.EDU_LABELS_KO].index(L.EDU_EN2KO[st.session_state["s_Education"]]))
        mar_ko = st.selectbox("결혼상태", [k for k, v in L.MARITAL_LABELS_KO],
                               index=[k for k, v in L.MARITAL_LABELS_KO].index(L.MARITAL_EN2KO[st.session_state["s_Marital_Status"]]))
        income = st.number_input("소득", 0, 1000000, key="s_Income", step=1000)
        kidhome = st.slider("자녀(어린이)", 0, 3, key="s_Kidhome")
        teenhome = st.slider("자녀(청소년)", 0, 3, key="s_Teenhome")
        recency = st.slider("구매경과일", 0, 100, key="s_Recency")
        deals = st.slider("할인구매횟수", 0, 20, key="s_NumDealsPurchases")
        web = st.slider("웹구매횟수", 0, 25, key="s_NumWebPurchases")
        catalog = st.slider("카탈로그구매횟수", 0, 25, key="s_NumCatalogPurchases")
        store = st.slider("매장구매횟수", 0, 25, key="s_NumStorePurchases")
        visits = st.slider("웹방문횟수", 0, 25, key="s_NumWebVisitsMonth")
        response = st.selectbox("최근캠페인반응", ["0", "1"],
                                 index=["0", "1"].index(st.session_state["s_Response"]))

        analyze_clicked = st.button("분석하기", type="primary", key="btn_analyze_single")

    with col_result:
        if analyze_clicked:
            customer = {
                "Year_Birth": year_birth,
                "Education": L.EDU_KO2EN.get(edu_ko, "Graduation"),
                "Marital_Status": L.MARITAL_KO2EN.get(mar_ko, "Married"),
                "Income": income,
                "Kidhome": kidhome,
                "Teenhome": teenhome,
                "Recency": recency,
                "NumDealsPurchases": deals,
                "NumWebPurchases": web,
                "NumCatalogPurchases": catalog,
                "NumStorePurchases": store,
                "NumWebVisitsMonth": visits,
                "Response": response,
            }
            r = L.analyze_customer(models, customer)
            p = r["prediction"]

            st.metric("VIP 여부", "⭐ VIP" if r["is_vip"] else "일반",
                      help=str(p["vip"]["probabilities"]))
            st.metric("추천 상품 카테고리", r["recommended_product"],
                      help=f"{p['interest']['prediction']} / {p['interest']['probabilities']}")
            st.metric("할인 민감성", p["discount"]["prediction"],
                      help=str(p["discount"]["probabilities"]))
            st.info(f"**프로모션 제안**: {r['promotion']}")
            if r["is_churn_risk"]:
                st.warning(f"🔻 이탈 위험 고객입니다 (구매경과일 {int(recency)}일)")
            st.metric("우선순위 점수", f"{r['priority_score']} / 100")
            st.caption(f"태그: {r['tags']}")
        else:
            st.info("왼쪽에 고객 정보를 입력하고 '분석하기'를 눌러주세요.")

# ---------------------------------------------------------------------------
# 탭 2 & 3. 여러 고객 비교 / 재유치 대상
# ---------------------------------------------------------------------------
TABLE_COLUMNS = [
    "이름", "Year_Birth", "Education", "Marital_Status", "Income", "Kidhome", "Teenhome",
    "Recency", "NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases",
    "NumStorePurchases", "NumWebVisitsMonth", "Response",
]
EXAMPLE_ROW = ["고객1", 1985, "Graduation", "Married", 65000, 0, 1, 10, 3, 5, 4, 6, 4, "1"]

if "multi_df" not in st.session_state:
    st.session_state.multi_df = pd.DataFrame([EXAMPLE_ROW], columns=TABLE_COLUMNS)
if "priority_result" not in st.session_state:
    st.session_state.priority_result = None
if "churn_result" not in st.session_state:
    st.session_state.churn_result = None


def run_multi_analysis(df: pd.DataFrame):
    results = []
    for _, row in df.iterrows():
        try:
            customer = {col: row[col] for col in TABLE_COLUMNS if col != "이름"}
            r = L.analyze_customer(models, customer)
            p = r["prediction"]
            results.append({
                "이름": row["이름"],
                "VIP여부": p["vip"]["prediction"],
                "추천상품": r["recommended_product"],
                "할인민감성": p["discount"]["prediction"],
                "프로모션": r["promotion"],
                "우선순위점수": r["priority_score"],
                "태그": r["tags"],
                "_is_vip": r["is_vip"],
                "_is_churn": r["is_churn_risk"],
            })
        except Exception as e:
            results.append({
                "이름": row.get("이름", "?"), "VIP여부": "오류", "추천상품": str(e),
                "할인민감성": "-", "프로모션": "-", "우선순위점수": 0, "태그": "-",
                "_is_vip": False, "_is_churn": False,
            })
    out = pd.DataFrame(results)
    out_sorted = out.sort_values(by=["_is_vip", "우선순위점수"], ascending=[False, False])
    priority_table = out_sorted.drop(columns=["_is_vip", "_is_churn"]).reset_index(drop=True)
    churn_table = out[out["_is_churn"]].drop(columns=["_is_vip", "_is_churn"]).reset_index(drop=True)
    return priority_table, churn_table


with tab2:
    st.markdown(
        "표에 직접 값을 입력하거나(행 추가 가능), 아래 랜덤 생성 버튼으로 샘플 고객을 채울 수 있어요.\n\n"
        "- Education 값: `2n Cycle`, `Basic`, `Graduation`, `Master`, `PhD`\n"
        "- Marital_Status 값: `Single`, `Married`, `Together`, `Divorced`, `Widow`\n"
        "- Response 값: `0` 또는 `1`"
    )

    n_random = st.number_input("랜덤 생성 인원수", 1, 50, 5, key="n_random")
    if st.button("🎲 랜덤 고객 여러 명 생성", key="btn_gen_random_rows"):
        rows = []
        for i in range(int(n_random)):
            c = L.random_customer()
            rows.append([f"고객{i+1}"] + [c[col] for col in TABLE_COLUMNS if col != "이름"])
        st.session_state.multi_df = pd.DataFrame(rows, columns=TABLE_COLUMNS)

    edited_df = st.data_editor(
        st.session_state.multi_df,
        num_rows="dynamic",
        use_container_width=True,
        key="multi_editor",
    )

    if st.button("전체 분석 & 우선순위 정렬", type="primary", key="btn_run_multi"):
        priority_table, churn_table = run_multi_analysis(edited_df)
        st.session_state.priority_result = priority_table
        st.session_state.churn_result = churn_table

    st.markdown("### 📊 우선순위 정렬 결과 (VIP 우선, 이후 점수순)")
    if st.session_state.priority_result is not None:
        st.dataframe(st.session_state.priority_result, use_container_width=True)
    else:
        st.caption("아직 분석을 실행하지 않았습니다.")

with tab3:
    st.markdown(f"구매경과일이 **{L.CHURN_THRESHOLD_DAYS}일 이상**인 고객을 재유치(휴면) 대상으로 표시합니다. "
                "'여러 고객 비교' 탭에서 분석을 실행하면 여기에도 함께 표시돼요.")
    if st.session_state.churn_result is not None:
        if len(st.session_state.churn_result) > 0:
            st.dataframe(st.session_state.churn_result, use_container_width=True)
        else:
            st.success("현재 재유치 대상 고객이 없습니다.")
    else:
        st.caption("아직 분석을 실행하지 않았습니다.")
