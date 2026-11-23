from __future__ import annotations

from datetime import date
import json
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize


TRADING_DAYS = 252
DEFAULT_TICKERS = ["AAPL", "SPY", "TLT", "GLD", "VNQ"]
DEFAULT_MARKET_WEIGHTS = [0.25, 0.35, 0.20, 0.10, 0.10]
RISK_TO_DELTA = {"保守型": 4.0, "穩健型": 2.5, "積極型": 1.2}
PREF_TO_BUCKET = {"極度保守": 0, "保守型": 0, "穩健型": 1, "積極型": 2, "極度積極": 2}


st.set_page_config(
    page_title="KYC × Black–Litterman 理財機器人",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    header[data-testid="stHeader"] { background: transparent !important; }
    .hero { background: linear-gradient(135deg, #0b1f3a, #164e63); padding: 2.2rem 2.5rem; border-radius: 22px; margin-bottom: 1.5rem; box-shadow: 0 10px 30px rgba(11,31,58,.15); }
    .hero h1 { color: #fff !important; margin: 0 0 .5rem; font-size: 2.25rem; }
    .hero p { color: #cbd5e1 !important; margin: 0; line-height: 1.6; }
    [data-testid="stMetric"] { border: 1px solid #e2e8f0; border-radius: 14px; padding: 12px; background: #fff; }
    [data-testid="stMetricLabel"] p { color: #475569; font-weight: 700; }
    .profile-card { border: 1px solid #dbeafe; background: #eff6ff; border-radius: 16px; padding: 1rem 1.2rem; height: 100%; }
    .feedback-card { border: 1px solid #e2e8f0; background: #fff; border-radius: 14px; padding: 1rem 1.1rem; min-height: 145px; box-shadow: 0 3px 12px rgba(15,23,42,.04); }
    .note { color: #64748b; font-size: .9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================== 資料與 BL 核心 ==============================
def parse_tickers(text: str) -> list[str]:
    result: list[str] = []
    for item in text.replace("，", ",").split(","):
        ticker = item.strip().upper()
        if ticker and ticker not in result:
            result.append(ticker)
    return result


def normalise_weights(values: Iterable[float], n_assets: int) -> np.ndarray:
    weights = np.asarray(list(values), dtype=float)
    if len(weights) != n_assets or not np.all(np.isfinite(weights)) or np.any(weights < 0):
        return np.ones(n_assets) / n_assets
    total = weights.sum()
    return weights / total if total > 0 else np.ones(n_assets) / n_assets


@st.cache_data(ttl=3600, show_spinner="正在下載並整理歷史價格資料…")
def download_prices(tickers: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    end_inclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(
        list(tickers), start=start, end=end_inclusive, interval="1d",
        auto_adjust=True, progress=False, threads=False, group_by="column",
    )
    if raw is None or raw.empty:
        raise ValueError("查不到資料，請確認日期區間或股票代號。")
    if isinstance(raw.columns, pd.MultiIndex):
        level_zero = set(raw.columns.get_level_values(0))
        level_one = set(raw.columns.get_level_values(1))
        if "Close" in level_zero:
            prices = raw["Close"].copy()
        elif "Close" in level_one:
            prices = raw.xs("Close", axis=1, level=1).copy()
        else:
            raise ValueError("下載結果找不到 Close 欄位。")
    elif "Close" in raw.columns:
        prices = raw[["Close"]].copy()
        prices.columns = [tickers[0]]
    else:
        raise ValueError("下載結果找不到 Close 欄位。")
    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices = prices.dropna(axis=1, how="all").dropna(how="any")
    prices.columns = [str(column).upper() for column in prices.columns]
    if prices.shape[1] < 2 or prices.shape[0] < 60:
        raise ValueError("有效資料不足。請至少使用兩檔有效標的，且日期區間需包含約 60 個交易日。")
    return prices


def build_statistics(prices: pd.DataFrame, risk_free_rate: float) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    returns = prices.pct_change(fill_method=None).dropna(how="any")
    if returns.shape[0] < 30:
        raise ValueError("有效日報酬資料不足，無法估計共變異數矩陣。")
    historical_excess = returns.mean().to_numpy() * TRADING_DAYS - risk_free_rate
    covariance = returns.cov().to_numpy() * TRADING_DAYS
    covariance = (covariance + covariance.T) / 2 + np.eye(len(covariance)) * 1e-10
    return returns, historical_excess, covariance


def make_views(
    tickers: list[str], absolute_asset: str, absolute_excess: float,
    relative_long: str, relative_short: str, relative_spread: float,
) -> tuple[np.ndarray, np.ndarray]:
    absolute = np.zeros(len(tickers))
    absolute[tickers.index(absolute_asset)] = 1.0
    relative = np.zeros(len(tickers))
    relative[tickers.index(relative_long)] = 1.0
    relative[tickers.index(relative_short)] = -1.0
    return np.vstack([absolute, relative]), np.array([absolute_excess, relative_spread])


def black_litterman_posterior(
    pi: np.ndarray, covariance: np.ndarray, p_matrix: np.ndarray,
    q_vector: np.ndarray, tau: float, confidences: Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    confidence = np.clip(np.asarray(list(confidences), dtype=float), 0.10, 0.95)
    view_variance = np.diag(p_matrix @ covariance @ p_matrix.T)
    omega = np.diag(((1 - confidence) / confidence) * tau * np.maximum(view_variance, 1e-12))
    tau_sigma = tau * covariance
    middle = p_matrix @ tau_sigma @ p_matrix.T + omega
    adjustment = tau_sigma @ p_matrix.T @ np.linalg.solve(middle, q_vector - p_matrix @ pi)
    return pi + adjustment, omega


def optimise_portfolio(expected_excess: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    n_assets = len(expected_excess)
    initial = np.ones(n_assets) / n_assets

    def objective(weights: np.ndarray) -> float:
        volatility = np.sqrt(max(float(weights @ covariance @ weights), 1e-14))
        return -float(weights @ expected_excess) / volatility

    result = minimize(
        objective, initial, method="SLSQP", bounds=[(0.0, 1.0)] * n_assets,
        constraints={"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        return initial
    weights = np.clip(result.x, 0.0, 1.0)
    return weights / weights.sum()


def portfolio_metrics(weights: np.ndarray, expected_excess: np.ndarray, covariance: np.ndarray, rf: float) -> dict[str, float]:
    excess_return = float(weights @ expected_excess)
    volatility = float(np.sqrt(max(weights @ covariance @ weights, 0.0)))
    return {"expected_return": rf + excess_return, "volatility": volatility, "sharpe": excess_return / volatility if volatility > 1e-12 else np.nan}


def calculate_model_results(
    market_weights: np.ndarray, historical_excess: np.ndarray, covariance: np.ndarray,
    delta: float, tau: float, p_matrix: np.ndarray, q_vector: np.ndarray,
    confidences: Iterable[float],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    pi = delta * covariance @ market_weights
    mu_bl, _ = black_litterman_posterior(pi, covariance, p_matrix, q_vector, tau, confidences)
    weights = {
        "Market Prior": market_weights,
        "傳統 MVO": optimise_portfolio(historical_excess, covariance),
        "Black–Litterman": optimise_portfolio(mu_bl, covariance),
    }
    return weights, pi, mu_bl


def sensitivity_frame(tickers: list[str], pi: np.ndarray, covariance: np.ndarray, p_matrix: np.ndarray, q_vector: np.ndarray, tau: float, confidence_two: float) -> pd.DataFrame:
    rows = []
    for confidence_one in np.linspace(0.10, 0.90, 17):
        mu_bl, _ = black_litterman_posterior(pi, covariance, p_matrix, q_vector, tau, [confidence_one, confidence_two])
        weights = optimise_portfolio(mu_bl, covariance)
        for ticker, weight in zip(tickers, weights):
            rows.append({"觀點 1 信心 (%)": confidence_one * 100, "資產": ticker, "BL 權重 (%)": weight * 100})
    return pd.DataFrame(rows)


# ============================== KYC 問卷與風險分級 ==============================
MCQ_LIST = [
    ("Q1 損失規避：投資單月下跌 20%，您的第一反應是？", ["全數認賠殺出，換回現金", "不動作，觀望基本面", "視為買點，逢低加碼"]),
    ("Q2 從眾傾向：新聞與朋友都在瘋狂討論熱門股，您會？", ["完全不碰，堅守原有配置", "研究財報，合理才少量買進", "怕錯過行情，立刻跟進買入"]),
    ("Q3 過度自信：您認為自己挑選標的的能力如何？", ["無法預測，只買大盤指數", "能挑選好股，也搭配大盤分散", "絕對有信心能擊敗大盤績效"]),
    ("Q4 心理帳戶：拿到一筆 10 萬元的意外獎金，您會？", ["全部存入銀行或買保本定存", "一半儲蓄、一半投入穩健資產", "全部投入股市追求高報酬"]),
    ("Q5 後悔厭惡：發現之前錯賣的股票後來大漲，您會？", ["不再碰該檔股票避免心痛", "重新評估目前的合理價位", "為了賺回來，立刻追高買回"]),
    ("Q6 近期偏誤：如果股市連續大漲三個月，您對未來預期是？", ["覺得快崩盤了，全面獲利了結", "保持既有投資節奏，不改變", "認為牛市確立，借錢或加碼"]),
    ("Q7 波動忍受：能忍受資產一年內最大縮水多少而不影響睡眠？", ["5% 以內", "10% ~ 15%", "20% 以上"]),
    ("Q8 熟悉度偏誤：選擇投資標的時，您傾向？", ["只買最熟悉的本土定存／金融股", "國內外知名大型股及 ETF", "積極尋找爆發力新興科技小股"]),
    ("Q9 耐心與短視：投資的股票三個月沒漲但大盤在漲，您會？", ["覺得不安心，賣掉換現金", "耐心等待，相信自己的研究", "覺得太慢，立刻換成熱門飆股"]),
    ("Q10 紀律自制力：每個月發薪水時，您的投資紀律是？", ["嚴格執行先儲蓄投資，剩下消費", "偶爾會因當月開銷大停止扣款", "常憑感覺隨意挪用投資金消費"]),
]


def call_cgu_llm(answers: dict, api_key: str, model_name: str, api_url: str) -> dict | None:
    system_prompt = """你是一位專業的行為金融學專家。請閱讀使用者回答，在五個維度給出 0~100 的數字評分與三段回饋。只回傳 JSON：
    {"radar_scores":{"損失承受力":0,"獨立判斷力":0,"適度自信度":0,"紀律自制力":0,"抗焦慮沉著":0},"feedback":{"current_emotional_state":"","behavioral_alert":"","personalized_advice":""}}"""
    user_prompt = "\n".join([
        f"Q1:{answers['q1_text']}", f"Q2:{answers['q2_text']}", f"Q3:{answers['q3_text']}", f"Q4:{answers['q4_text']}",
        f"負債:{answers['debt_status']}", f"流動性:{answers['liquidity_need']}",
    ])
    try:
        response = requests.post(
            api_url,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            json={"model": model_name, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]},
            timeout=20,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(item.get("text", "") for item in content if isinstance(item, dict))
        content = str(content).replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except Exception as error:
        st.warning(f"AI 問卷分析無法使用，已改用中性分數 50 分：{error}")
        return None


def radar_from_mcq(values: list[int]) -> dict[str, float]:
    # 完整保留 legacy 版本的五構面計分邏輯。
    return {
        "損失承受力": (values[0] + values[6]) * 25,
        "獨立判斷力": ((2 - values[1]) + (2 - values[4])) * 25,
        "適度自信度": (values[2] + values[7]) * 25,
        "紀律自制力": ((2 - values[8]) + (2 - values[9])) * 25,
        "抗焦慮沉著": ((2 - values[3]) + (2 - values[5])) * 25,
    }


def classify_investor(answers: dict, radar_scores: dict[str, float]) -> tuple[str, str, str, str]:
    # 完整保留 legacy 版本的主觀偏好、問卷與財務硬限制分級邏輯。
    level = PREF_TO_BUCKET[answers["risk_pref"]]
    if answers["mode"] == "選擇題":
        behavioural_average = sum(answers["q_scores"]) / 10.0
    else:
        ai_average = sum(radar_scores.values()) / 5
        behavioural_average = 0.0 if ai_average < 40 else 2.0 if ai_average > 70 else 1.0

    if behavioural_average <= 0.6 and level > 0:
        level -= 1
    elif behavioural_average >= 1.4 and level < 2:
        level += 1

    debt_text = answers["debt_status"].replace(" ", "")
    liquidity_text = answers["liquidity_need"].replace(" ", "")
    red_debt = any(key in debt_text for key in ["信貸", "卡債", "高利", "重", "壓力大", "還不起"])
    yellow_debt = any(key in debt_text for key in ["房貸", "車貸", "學貸", "貸款"])
    red_liquidity = any(key in liquidity_text for key in ["急用", "醫療", "結婚", "買房", "購屋", "頭期款", "一年", "半年", "近期"])

    status_color, title, message = "green", "綠燈 · 整體條件吻合", "基礎風險需求與客觀財務條件整體符合，可依計畫執行。"
    if answers["horizon_years"] < 5 or red_liquidity or red_debt:
        level, status_color, title, message = 0, "red", "紅燈 · 觸發財務硬限制", "偵測到投資期限過短或有沉重負債／近期大額開銷，系統已降級為保守型以保護本金。"
    elif answers["investable_pct"] < 20 or yellow_debt:
        level = min(level, 1)
        status_color, title, message = "yellow", "黃燈 · 需留意現金流", "偵測到財務空間較緊繃或背負中長天期貸款，系統建議最高配置為穩健型。"
    return {0: "保守型", 1: "穩健型", 2: "積極型"}[level], status_color, title, message


def local_feedback(radar_scores: dict[str, float], status_color: str, final_label: str) -> tuple[str, str, str]:
    if radar_scores["抗焦慮沉著"] >= 60 and radar_scores["損失承受力"] >= 60:
        emotion = "目前情緒狀態穩定且具備良好風險承受力，面對市場波動應能保持客觀理性。"
    elif radar_scores["抗焦慮沉著"] < 50:
        emotion = "對市場波動的焦慮感較高，容易因短期虧損產生恐慌或不安全感。"
    else:
        emotion = "情緒狀態偏向保守防禦，可能對潛在資金回撤感到一定程度的心理壓力。"
    lowest = min(radar_scores, key=radar_scores.get)
    alerts = {
        "獨立判斷力": "從眾傾向較高，請避免因市場熱潮或親友推薦而盲目追高。",
        "紀律自制力": "財務紀律可能較鬆散，建議設定自動扣款與定期定額機制。",
        "適度自信度": "可能有過度自信傾向，請避免單筆重押單一標的並落實分散。",
    }
    alert = alerts.get(lowest, "面對市場下跌，請避免因短線恐慌而改變長期投資計畫。")
    if status_color == "red":
        advice = "財務硬限制亮紅燈，首要任務是降低負債或補足緊急預備金。"
    elif status_color == "yellow":
        advice = "現金流略顯緊繃，建議先確保 6 個月生活預備金，再逐步投入。"
    elif final_label == "積極型":
        advice = "可承受較高風險，但仍應搭配跨資產分散，避免單一標的集中。"
    elif final_label == "穩健型":
        advice = "以股債與跨資產平衡為主，透過 BL 觀點調整而非追逐短期熱門標的。"
    else:
        advice = "防禦優先，建議控制波動並保留足夠現金部位。"
    return emotion, alert, advice


# ============================== 圖表 ==============================
def plot_layout(fig, height: int = 430):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=20, r=20, t=55, b=20), legend_title_text="", hovermode="x unified")
    return fig


def plot_radar(scores: dict[str, float]):
    categories = list(scores.keys())
    values = list(scores.values())
    fig = go.Figure(go.Scatterpolar(r=values + values[:1], theta=categories + categories[:1], fill="toself", line=dict(color="#0f766e", width=3), fillcolor="rgba(15,118,110,.16)", marker=dict(color="#2563eb", size=8)))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100], gridcolor="#e2e8f0"), bgcolor="#f8fafc"), showlegend=False, height=340, margin=dict(l=40, r=40, t=25, b=25), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def plot_correlation_heatmap(corr: pd.DataFrame):
    fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, text=np.round(corr.values, 2), texttemplate="%{text}", colorscale="RdBu_r", zmin=-1, zmax=1, hovertemplate="資產 X: %{x}<br>資產 Y: %{y}<br>相關係數: %{z:.2f}<extra></extra>"))
    fig.update_layout(height=380, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor="rgba(0,0,0,0)")
    fig.update_yaxes(autorange="reversed")
    return fig


# ============================== 側邊欄模型設定 ==============================
with st.sidebar:
    st.header("⚙️ 模型設定")
    st.subheader("資產與資料區間")
    ticker_text = st.text_input("美股代號（以逗號分隔）", value=", ".join(DEFAULT_TICKERS))
    tickers = parse_tickers(ticker_text)
    if len(tickers) < 2:
        st.error("至少需要 2 檔有效標的，才能建立相對觀點。")
        st.stop()
    start_date = st.date_input("資料開始日", value=date(2021, 10, 1), key="start_date")
    end_date = st.date_input("資料結束日", value=date(2026, 10, 1), key="end_date")
    if start_date >= end_date:
        st.error("資料開始日必須早於資料結束日。")
        st.stop()

    st.subheader("BL 核心參數")
    risk_free_rate = st.slider("無風險利率 Rf", 0.0, 0.10, 0.04, 0.005, format="%.1f%%")
    tau = st.slider("先驗不確定性尺度 τ", 0.01, 0.50, 0.05, 0.01, format="%.2f")
    profile_for_delta = st.session_state.get("risk_profile")
    if profile_for_delta:
        delta = float(profile_for_delta["delta"])
        st.metric("問卷連動風險趨避係數 δ", f"{delta:.1f}")
        st.caption(f"{profile_for_delta['final_label']} → δ={delta:.1f}（由 KYC 自動連動）")
    else:
        delta = 2.5
        st.info("完成下方 KYC 問卷後，δ 會自動更新。")

    st.subheader("市場基準權重")
    market_weight_text = st.text_input("依序輸入權重", value=", ".join(f"{value:.2f}" for value in DEFAULT_MARKET_WEIGHTS), help="權重會自動正規化；數量不符時使用等權重。")
    try:
        raw_market_weights = [float(value.strip()) for value in market_weight_text.replace("，", ",").split(",") if value.strip()]
    except ValueError:
        raw_market_weights = []
    market_weights = normalise_weights(raw_market_weights, len(tickers))
    st.caption("已自動正規化為總和 100%。" if len(raw_market_weights) == len(tickers) else "權重數量與資產數量不符，目前使用等權重。")

    with st.expander("投資人觀點與信心旋鈕", expanded=True):
        absolute_default = "AAPL" if "AAPL" in tickers else tickers[0]
        absolute_asset = st.selectbox("觀點 1｜絕對觀點資產", tickers, index=tickers.index(absolute_default))
        absolute_excess = st.slider("觀點 1｜預期年化超額報酬", -0.20, 0.30, 0.10, 0.01, format="%.0f%%")
        confidence_one = st.slider("觀點 1｜信心水準 c1", 10, 95, 70, 5, format="%d%%") / 100
        relative_default_long = "SPY" if "SPY" in tickers else tickers[0]
        relative_default_short = "TLT" if "TLT" in tickers else tickers[1]
        relative_long = st.selectbox("觀點 2｜看多資產 A", tickers, index=tickers.index(relative_default_long))
        short_options = [ticker for ticker in tickers if ticker != relative_long]
        if relative_default_short not in short_options:
            relative_default_short = short_options[0]
        relative_short = st.selectbox("觀點 2｜看空資產 B", short_options, index=short_options.index(relative_default_short))
        relative_spread = st.slider("觀點 2｜預期超額報酬差距", -0.20, 0.30, 0.03, 0.01, format="%.0f%%")
        confidence_two = st.slider("觀點 2｜信心水準 c2", 10, 95, 50, 5, format="%d%%") / 100

    with st.expander("開放式問答 AI 設定（選用）", expanded=False):
        api_url = st.text_input("API Base URL", value="https://air.cgu.edu.tw/cgullmapi/v1/chat/completions")
        model_name = st.text_input("模型名稱", value="gpt-5.6-luna")
        api_key = st.text_input("API Key", type="password", placeholder="開放式問答模式才需要")


# ============================== 主畫面：KYC ==============================
st.markdown("<div class='hero'><h1>📊 理財機器人：KYC × Black–Litterman 配置平台</h1><p>先用客戶風險屬性問卷建立風險畫像，再將風險屬性自動連動到 BL 的 δ，結合個人觀點產出資產配置。</p></div>", unsafe_allow_html=True)

st.header("一、理財機器人客戶風險屬性問卷（KYC / Questionnaire）")
with st.expander("填寫完整 KYC 問卷", expanded="risk_profile" not in st.session_state):
    st.subheader("Stage 01｜傳統風險評估")
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("年齡", min_value=18, max_value=120, value=32, key="kyc_age")
        liquid_assets = st.number_input("流動資產（萬）", min_value=0, value=80, key="kyc_liquid_assets")
        horizon_value = st.number_input("投資期限", min_value=1, value=8, key="kyc_horizon_value")
    with c2:
        income = st.number_input("年收入（萬）", min_value=0, value=120, key="kyc_income")
        real_estate = st.number_input("不動產（萬）", min_value=0, value=0, key="kyc_real_estate")
        horizon_unit = st.selectbox("期限單位", ["年", "月"], key="kyc_horizon_unit")
    risk_pref = st.select_slider("主觀風險偏好", options=["極度保守", "保守型", "穩健型", "積極型", "極度積極"], value="穩健型", key="kyc_risk_pref")

    st.subheader("Stage 02｜行為與情緒評估")
    mode = st.radio("請選擇問卷填寫模式", ["選擇題（快速適配）", "開放式問答（AI 深度分析）"], horizontal=True, key="kyc_mode")
    q_scores = [1] * 10
    q1_text = q2_text = q3_text = q4_text = ""
    if mode == "選擇題（快速適配）":
        q_col_1, q_col_2 = st.columns(2)
        for index, (question, options) in enumerate(MCQ_LIST):
            with q_col_1 if index < 5 else q_col_2:
                answer = st.radio(question, options, key=f"kyc_q_{index + 1}")
                q_scores[index] = options.index(answer)
    else:
        st.caption("若未提供 API Key，系統會以五構面中性分數 50 分完成分級，仍會套用財務硬限制。")
        q1_text = st.text_area("Q1 損失規避：若市場下跌 20%，您會立刻停損還是繼續加碼？為什麼？", key="kyc_text_1")
        q2_text = st.text_area("Q2 從眾傾向：看到朋友買熱門股大賺，您會改變策略跟進嗎？", key="kyc_text_2")
        q3_text = st.text_area("Q3 過度自信：您認為自己過去的投資績效是靠實力還是運氣？", key="kyc_text_3")
        q4_text = st.text_area("Q4 自制力：拿到意外獎金，您會立刻消費還是依計畫投資？", key="kyc_text_4")

    st.subheader("Stage 03｜財務條件與極限")
    c3, c4 = st.columns(2)
    with c3:
        expenses = st.number_input("每月支出（萬）", min_value=0.0, value=4.5, key="kyc_expenses")
    with c4:
        inv_method = st.selectbox("預計投入方式", ["每月定期定額", "單筆現金 All-in", "單筆打底 + 定期定額"], key="kyc_inv_method")
    inv_percent = st.slider("可投資資金佔淨資產比例（%）", 0, 100, 40, key="kyc_inv_percent")
    debt_status = st.text_input("負債狀況與結構", value="無負債", placeholder="例如：每月房貸 3 萬，無信貸", key="kyc_debt")
    liquidity_need = st.text_input("短期流動性需求（一年內）", value="無需求", placeholder="例如：預計兩年後需要結婚基金，或目前無需求", key="kyc_liquidity")

    confirm_kyc = st.button("✅ 確認問卷結果並同步 BL δ", type="primary")

if confirm_kyc:
    answers = {
        "mode": "選擇題" if mode.startswith("選擇題") else "開放式",
        "horizon_years": horizon_value if horizon_unit == "年" else horizon_value / 12,
        "risk_pref": risk_pref, "q_scores": q_scores,
        "q1_text": q1_text, "q2_text": q2_text, "q3_text": q3_text, "q4_text": q4_text,
        "investable_pct": inv_percent, "debt_status": debt_status, "liquidity_need": liquidity_need,
    }
    ai_feedback = None
    if answers["mode"] == "選擇題":
        radar_scores = radar_from_mcq(answers["q_scores"])
    elif api_key:
        with st.spinner("正在分析開放式問答…"):
            ai_result = call_cgu_llm(answers, api_key, model_name, api_url)
        radar_scores = (ai_result or {}).get("radar_scores", {})
        ai_feedback = (ai_result or {}).get("feedback", {})
        if not radar_scores:
            radar_scores = {"損失承受力": 50, "獨立判斷力": 50, "適度自信度": 50, "紀律自制力": 50, "抗焦慮沉著": 50}
    else:
        radar_scores = {"損失承受力": 50, "獨立判斷力": 50, "適度自信度": 50, "紀律自制力": 50, "抗焦慮沉著": 50}
        st.info("未輸入 API Key，開放式問答暫以中性分數 50 分估算。")

    radar_scores = {key: float(np.clip(value, 0, 100)) for key, value in radar_scores.items()}
    final_label, status_color, status_title, status_msg = classify_investor(answers, radar_scores)
    st.session_state["risk_profile"] = {
        "answers": answers, "radar_scores": radar_scores, "ai_feedback": ai_feedback,
        "final_label": final_label, "delta": RISK_TO_DELTA[final_label],
        "risk_score": float(np.mean(list(radar_scores.values()))),
        "status_color": status_color, "status_title": status_title, "status_msg": status_msg,
    }
    st.rerun()


profile = st.session_state.get("risk_profile")
if not profile:
    st.info("請完成上方 KYC 問卷並按下「確認問卷結果並同步 BL δ」，系統才會產生個人化配置。")
    st.stop()


# ============================== 主畫面：風險畫像 ==============================
st.header("二、問卷風險畫像與 BL 參數連動")
profile_cols = st.columns(4)
profile_cols[0].metric("風險承受度分數", f"{profile['risk_score']:.1f} / 100")
profile_cols[1].metric("客戶風險屬性", profile["final_label"])
profile_cols[2].metric("BL 風險趨避係數 δ", f"{profile['delta']:.1f}")
profile_cols[3].metric("問卷填寫模式", "選擇題" if profile["answers"]["mode"] == "選擇題" else "開放式")
st.info(f"連動關係：問卷與財務條件判定為「{profile['final_label']}」→ Black–Litterman δ = {profile['delta']:.1f}。δ 越高代表風險趨避程度越高；此值不再由獨立滑桿覆寫。")

status_styles = {
    "green": ("#ecfdf5", "#a7f3d0", "#047857"),
    "yellow": ("#fffbeb", "#fde68a", "#b45309"),
    "red": ("#fff1f2", "#fecaca", "#be123c"),
}
bg, border, text_color = status_styles[profile["status_color"]]
st.markdown(f"<div style='background:{bg};border:1px solid {border};border-radius:14px;padding:1rem 1.2rem;margin:.7rem 0 1.2rem'><b style='color:{text_color}'>{profile['status_title']}</b><br><span style='color:{text_color}'>{profile['status_msg']}</span></div>", unsafe_allow_html=True)

radar_col, feedback_col = st.columns([1, 1.8])
with radar_col:
    st.plotly_chart(plot_radar(profile["radar_scores"]), width="stretch", config={"displayModeBar": False})
with feedback_col:
    ai_feedback = profile.get("ai_feedback") or {}
    if profile["answers"]["mode"] == "開放式" and ai_feedback:
        emotion = ai_feedback.get("current_emotional_state", "無資料")
        alert = ai_feedback.get("behavioral_alert", "無資料")
        advice = ai_feedback.get("personalized_advice", "無資料")
    else:
        emotion, alert, advice = local_feedback(profile["radar_scores"], profile["status_color"], profile["final_label"])
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown(f"<div class='feedback-card'><b style='color:#0f766e'>🧠 情緒狀態</b><p>{emotion}</p></div>", unsafe_allow_html=True)
    with f2:
        st.markdown(f"<div class='feedback-card'><b style='color:#d97706'>⚠️ 行為提醒</b><p>{alert}</p></div>", unsafe_allow_html=True)
    with f3:
        st.markdown(f"<div class='feedback-card'><b style='color:#2563eb'>🎯 投資建議</b><p>{advice}</p></div>", unsafe_allow_html=True)


# ============================== 主畫面：BL / MVO 看板 ==============================
st.header("三、結合問卷 δ 與個人觀點的 BL / MVO 投資組合")
try:
    prices = download_prices(tuple(tickers), str(start_date), str(end_date))
    valid_tickers = list(prices.columns)
    missing = [ticker for ticker in tickers if ticker not in valid_tickers]
    if missing:
        st.warning(f"以下代號沒有可用資料，已自動排除：{', '.join(missing)}")
    returns, historical_excess, covariance = build_statistics(prices[valid_tickers], risk_free_rate)
    valid_market_weights = normalise_weights([market_weights[tickers.index(ticker)] for ticker in valid_tickers], len(valid_tickers))
    p_matrix, q_vector = make_views(
        valid_tickers,
        absolute_asset if absolute_asset in valid_tickers else valid_tickers[0], absolute_excess,
        relative_long if relative_long in valid_tickers else valid_tickers[0],
        relative_short if relative_short in valid_tickers else valid_tickers[1], relative_spread,
    )
    weights, pi, mu_bl = calculate_model_results(valid_market_weights, historical_excess, covariance, profile["delta"], tau, p_matrix, q_vector, [confidence_one, confidence_two])
except Exception as error:
    st.error(f"資料下載或模型計算失敗：{error}")
    st.stop()

st.caption(f"資料期間：{prices.index.min().date()} 至 {prices.index.max().date()}｜有效交易日：{len(returns):,}｜δ={profile['delta']:.1f}｜有效資產：{', '.join(valid_tickers)}")
expected_excess_by_model = {"Market Prior": pi, "傳統 MVO": historical_excess, "Black–Litterman": mu_bl}
metrics = {model: portfolio_metrics(model_weights, expected_excess_by_model[model], covariance, risk_free_rate) for model, model_weights in weights.items()}

metric_cols = st.columns(3)
for column, model in zip(metric_cols, ["Market Prior", "傳統 MVO", "Black–Litterman"]):
    with column:
        st.markdown(f"**{model}**")
        st.metric("年化預期報酬率", f"{metrics[model]['expected_return']:.2%}")
        st.metric("年化波動度", f"{metrics[model]['volatility']:.2%}")
        st.metric("夏普比率", f"{metrics[model]['sharpe']:.2f}")

st.subheader("資產配置權重比較")
weight_rows = [{"模型": model, "資產": ticker, "權重 (%)": weight * 100} for model, values in weights.items() for ticker, weight in zip(valid_tickers, values)]
weight_fig = px.bar(pd.DataFrame(weight_rows), x="資產", y="權重 (%)", color="模型", barmode="group", text_auto=".1f", color_discrete_sequence=["#64748b", "#f59e0b", "#2563eb"])
weight_fig.update_yaxes(title="權重 (%)", ticksuffix="%")
st.plotly_chart(plot_layout(weight_fig), width="stretch", config={"displayModeBar": False})

st.subheader("預期報酬率演變")
return_data = pd.DataFrame({"資產": valid_tickers, "歷史平均報酬": historical_excess + risk_free_rate, "市場隱含報酬 π": pi + risk_free_rate, "BL 後驗報酬 μ_BL": mu_bl + risk_free_rate}).melt(id_vars="資產", var_name="估計方式", value_name="年化報酬")
return_fig = px.bar(return_data, x="資產", y="年化報酬", color="估計方式", barmode="group", text_auto=".1%", color_discrete_sequence=["#94a3b8", "#f59e0b", "#2563eb"])
return_fig.update_yaxes(title="年化報酬率", tickformat=".1%")
st.plotly_chart(plot_layout(return_fig), width="stretch", config={"displayModeBar": False})

st.subheader("觀點 1 信心水準敏感度分析")
sensitivity = sensitivity_frame(valid_tickers, pi, covariance, p_matrix, q_vector, tau, confidence_two)
sensitivity_fig = px.line(sensitivity, x="觀點 1 信心 (%)", y="BL 權重 (%)", color="資產", markers=True, color_discrete_sequence=px.colors.qualitative.Safe)
sensitivity_fig.update_xaxes(title="觀點 1 信心水準", ticksuffix="%", range=[10, 90])
sensitivity_fig.update_yaxes(title="BL 最適權重", ticksuffix="%")
st.plotly_chart(plot_layout(sensitivity_fig, height=470), width="stretch", config={"displayModeBar": False})

st.subheader("完整數據明細")
detail = pd.DataFrame({"資產": valid_tickers, "歷史平均超額報酬": historical_excess, "市場隱含超額報酬 π": pi, "BL 後驗超額報酬 μ_BL": mu_bl, "Market Prior 權重": weights["Market Prior"], "傳統 MVO 權重": weights["傳統 MVO"], "BL 權重": weights["Black–Litterman"]})
display_detail = detail.copy()
for column in display_detail.columns[1:]:
    display_detail[column] = display_detail[column].map(lambda value: f"{value:.2%}")
st.dataframe(display_detail, width="stretch", hide_index=True)
st.download_button("⬇️ 下載 CSV 明細", data=detail.to_csv(index=False).encode("utf-8-sig"), file_name="black_litterman_portfolio_detail.csv", mime="text/csv")

with st.expander("模型說明", expanded=False):
    st.markdown("""
    - KYC 問卷先以主觀偏好、行為問卷與財務硬限制判定保守型／穩健型／積極型。
    - 風險屬性會自動連動 BL：保守型 δ=4.0、穩健型 δ=2.5、積極型 δ=1.2。
    - 市場均衡超額報酬：`π = δ × Σ × w_mkt`；信心不確定性：`ω² = ((1-c)/c) × (τ × PΣPᵀ)`。
    - 傳統 MVO 使用歷史平均超額報酬；BL 使用結合問卷 δ 與投資人觀點後的 μ_BL。
    - 三種組合皆限制權重總和為 1 且禁止放空。
    """)
