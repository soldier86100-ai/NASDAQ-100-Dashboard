import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import numpy as np

st.set_page_config(layout="wide", page_title="NASDAQ 100 Pro Market Dashboard")

# ==========================================
# 1. 核心數據定義 (NASDAQ 100)
# ==========================================

# 您提供的成分股清單 (含權重)
NASDAQ_CONSTITUENTS = [
    {'Symbol': 'NVDA', 'Company': 'NVIDIA Corp', 'Allocation': 0.1015},
    {'Symbol': 'AAPL', 'Company': 'Apple Inc', 'Allocation': 0.084},
    {'Symbol': 'MSFT', 'Company': 'Microsoft Corp', 'Allocation': 0.079},
    {'Symbol': 'AVGO', 'Company': 'Broadcom Inc', 'Allocation': 0.0585},
    {'Symbol': 'AMZN', 'Company': 'Amazon.com Inc', 'Allocation': 0.0556},
    {'Symbol': 'GOOGL', 'Company': 'Alphabet Inc Class A', 'Allocation': 0.0354},
    {'Symbol': 'TSLA', 'Company': 'Tesla Inc', 'Allocation': 0.0341},
    {'Symbol': 'GOOG', 'Company': 'Alphabet Inc Class C', 'Allocation': 0.0331},
    {'Symbol': 'META', 'Company': 'Meta Platforms Inc', 'Allocation': 0.0288},
    {'Symbol': 'NFLX', 'Company': 'Netflix Inc', 'Allocation': 0.0246},
    {'Symbol': 'PLTR', 'Company': 'Palantir Technologies Inc', 'Allocation': 0.0227},
    {'Symbol': 'COST', 'Company': 'Costco Wholesale Corp', 'Allocation': 0.021},
    {'Symbol': 'AMD', 'Company': 'Advanced Micro Devices Inc', 'Allocation': 0.0204},
    {'Symbol': 'CSCO', 'Company': 'Cisco Systems Inc', 'Allocation': 0.0147},
    {'Symbol': 'MU', 'Company': 'Micron Technology Inc', 'Allocation': 0.0146},
    {'Symbol': 'TMUS', 'Company': 'T-Mobile US Inc', 'Allocation': 0.012},
    {'Symbol': 'LRCX', 'Company': 'Lam Research Corp', 'Allocation': 0.0109},
    {'Symbol': 'ISRG', 'Company': 'Intuitive Surgical Inc', 'Allocation': 0.0107},
    {'Symbol': 'APP', 'Company': 'AppLovin Corp', 'Allocation': 0.0103},
    {'Symbol': 'LIN', 'Company': 'Linde PLC', 'Allocation': 0.0102},
    {'Symbol': 'PEP', 'Company': 'PepsiCo Inc', 'Allocation': 0.0101},
    {'Symbol': 'SHOP', 'Company': 'Shopify Inc', 'Allocation': 0.01},
    {'Symbol': 'AMAT', 'Company': 'Applied Materials Inc', 'Allocation': 0.0097},
    {'Symbol': 'QCOM', 'Company': 'QUALCOMM Inc', 'Allocation': 0.0096},
    {'Symbol': 'INTU', 'Company': 'Intuit Inc', 'Allocation': 0.0094},
    {'Symbol': 'AMGN', 'Company': 'Amgen Inc', 'Allocation': 0.009},
    {'Symbol': 'INTC', 'Company': 'Intel Corp', 'Allocation': 0.0087},
    {'Symbol': 'KLAC', 'Company': 'KLA Corp', 'Allocation': 0.0083},
    {'Symbol': 'BKNG', 'Company': 'Booking Holdings Inc', 'Allocation': 0.0083},
    {'Symbol': 'TXN', 'Company': 'Texas Instruments Inc', 'Allocation': 0.0076},
    {'Symbol': 'GILD', 'Company': 'Gilead Sciences Inc', 'Allocation': 0.0076},
    {'Symbol': 'PANW', 'Company': 'Palo Alto Networks Inc', 'Allocation': 0.0075},
    {'Symbol': 'CRWD', 'Company': 'Crowdstrike Holdings Inc', 'Allocation': 0.0072},
    {'Symbol': 'ADBE', 'Company': 'Adobe Inc', 'Allocation': 0.0072},
    {'Symbol': 'HON', 'Company': 'Honeywell International Inc', 'Allocation': 0.0064},
    {'Symbol': 'ADI', 'Company': 'Analog Devices Inc', 'Allocation': 0.0059},
    {'Symbol': 'CEG', 'Company': 'Constellation Energy Corp', 'Allocation': 0.0058},
    {'Symbol': 'VERX', 'Company': 'Vertex Pharmaceuticals Inc', 'Allocation': 0.0056},
    {'Symbol': 'MELI', 'Company': 'MercadoLibre Inc', 'Allocation': 0.0055},
    {'Symbol': 'ADP', 'Company': 'Automatic Data Processing Inc', 'Allocation': 0.0053},
    {'Symbol': 'CMCSA', 'Company': 'Comcast Corp', 'Allocation': 0.0052},
    {'Symbol': 'SBUX', 'Company': 'Starbucks Corp', 'Allocation': 0.005},
    {'Symbol': 'PDD', 'Company': 'PDD Holdings Inc', 'Allocation': 0.0048},
    {'Symbol': 'CDNS', 'Company': 'Cadence Design Systems Inc', 'Allocation': 0.0046},
    {'Symbol': 'ASML', 'Company': 'ASML Holding NV', 'Allocation': 0.0045},
    {'Symbol': 'ORLY', 'Company': "O\\'Reilly Automotive Inc", 'Allocation': 0.0043},
    {'Symbol': 'DASH', 'Company': 'DoorDash Inc', 'Allocation': 0.0042},
    {'Symbol': 'MRVL', 'Company': 'Marvell Technology Inc', 'Allocation': 0.0042},
    {'Symbol': 'MAR', 'Company': 'Marriott International Inc/MD', 'Allocation': 0.0041},
    {'Symbol': 'CTAS', 'Company': 'Cintas Corp', 'Allocation': 0.0039},
    {'Symbol': 'SNPS', 'Company': 'Synopsys Inc', 'Allocation': 0.0038},
    {'Symbol': 'MDLZ', 'Company': 'Mondelez International Inc', 'Allocation': 0.0038},
    {'Symbol': 'MNST', 'Company': 'Monster Beverage Corp', 'Allocation': 0.0035},
    {'Symbol': 'REGN', 'Company': 'Regeneron Pharmaceuticals Inc', 'Allocation': 0.0035},
    {'Symbol': 'AEP', 'Company': 'American Electric Power Co Inc', 'Allocation': 0.0034},
    {'Symbol': 'CSX', 'Company': 'CSX Corp', 'Allocation': 0.0034},
    {'Symbol': 'DDOG', 'Company': 'Datadog Inc', 'Allocation': 0.0033},
    {'Symbol': 'ADSK', 'Company': 'Autodesk Inc', 'Allocation': 0.0033},
    {'Symbol': 'PYPL', 'Company': 'PayPal Holdings Inc', 'Allocation': 0.0033},
    {'Symbol': 'FTNT', 'Company': 'Fortinet Inc', 'Allocation': 0.0033},
    {'Symbol': 'SMCI', 'Company': 'Strategy Inc', 'Allocation': 0.0033},
    {'Symbol': 'TRI', 'Company': 'Thomson Reuters Corp', 'Allocation': 0.0032},
    {'Symbol': 'WBD', 'Company': 'Warner Bros Discovery Inc', 'Allocation': 0.0029},
    {'Symbol': 'IDXX', 'Company': 'IDEXX Laboratories Inc', 'Allocation': 0.0029},
    {'Symbol': 'ROST', 'Company': 'Ross Stores Inc', 'Allocation': 0.0027},
    {'Symbol': 'PCAR', 'Company': 'PACCAR Inc', 'Allocation': 0.0027},
    {'Symbol': 'NXPI', 'Company': 'NXP Semiconductors NV', 'Allocation': 0.0027},
    {'Symbol': 'ABNB', 'Company': 'Airbnb Inc', 'Allocation': 0.0027},
    {'Symbol': 'AZN', 'Company': 'AstraZeneca PLC', 'Allocation': 0.0027},
    {'Symbol': 'ZS', 'Company': 'Zscaler Inc', 'Allocation': 0.0026},
    {'Symbol': 'EA', 'Company': 'Electronic Arts Inc', 'Allocation': 0.0026},
    {'Symbol': 'WDAY', 'Company': 'Workday Inc', 'Allocation': 0.0026},
    {'Symbol': 'ROP', 'Company': 'Roper Technologies Inc', 'Allocation': 0.0025},
    {'Symbol': 'BKR', 'Company': 'Baker Hughes Co', 'Allocation': 0.0025},
    {'Symbol': 'FAST', 'Company': 'Fastenal Co', 'Allocation': 0.0025},
    {'Symbol': 'XEL', 'Company': 'Xcel Energy Inc', 'Allocation': 0.0025},
    {'Symbol': 'EXC', 'Company': 'Exelon Corp', 'Allocation': 0.0024},
    {'Symbol': 'AXON', 'Company': 'Axon Enterprise Inc', 'Allocation': 0.0024},
    {'Symbol': 'TTWO', 'Company': 'Take-Two Interactive Software Inc', 'Allocation': 0.0022},
    {'Symbol': 'FANG', 'Company': 'Diamondback Energy Inc', 'Allocation': 0.0022},
    {'Symbol': 'CCEP', 'Company': 'Coca-Cola Europacific Partners PLC', 'Allocation': 0.0022},
    {'Symbol': 'PAYX', 'Company': 'Paychex Inc', 'Allocation': 0.0021},
    {'Symbol': 'CPRT', 'Company': 'Copart Inc', 'Allocation': 0.0021},
    {'Symbol': 'CTSH', 'Company': 'Cognizant Technology Solutions Corp', 'Allocation': 0.0018},
    {'Symbol': 'KDP', 'Company': 'Keurig Dr Pepper Inc', 'Allocation': 0.0018},
    {'Symbol': 'GEHC', 'Company': 'GE HealthCare Technologies Inc', 'Allocation': 0.0017},
    {'Symbol': 'MCHP', 'Company': 'Microchip Technology Inc', 'Allocation': 0.0015},
    {'Symbol': 'VRSK', 'Company': 'Verisk Analytics Inc', 'Allocation': 0.0015},
    {'Symbol': 'ODFL', 'Company': 'Old Dominion Freight Line Inc', 'Allocation': 0.0015},
    {'Symbol': 'KHC', 'Company': 'Kraft Heinz Co/The', 'Allocation': 0.0015},
    {'Symbol': 'CHTR', 'Company': 'Charter Communications Inc', 'Allocation': 0.0015},
    {'Symbol': 'CSGP', 'Company': 'CoStar Group Inc', 'Allocation': 0.0015},
    {'Symbol': 'TEAM', 'Company': 'Atlassian Corp', 'Allocation': 0.0014},
    {'Symbol': 'BIIB', 'Company': 'Biogen Inc', 'Allocation': 0.0012},
    {'Symbol': 'DXCM', 'Company': 'Dexcom Inc', 'Allocation': 0.0011},
    {'Symbol': 'ARM', 'Company': 'ARM Holdings PLC', 'Allocation': 0.0011},
    {'Symbol': 'ON', 'Company': 'ON Semiconductor Corp', 'Allocation': 0.001},
    {'Symbol': 'LULU', 'Company': 'Lululemon Athletica Inc', 'Allocation': 0.001},
    {'Symbol': 'TTD', 'Company': 'Trade Desk Inc/The', 'Allocation': 0.001},
    {'Symbol': 'GFS', 'Company': 'GLOBALFOUNDRIES Inc', 'Allocation': 0.001},
    {'Symbol': 'CDW', 'Company': 'CDW Corp/DE', 'Allocation': 0.001},
]

@st.cache_data(ttl=3600)
def parse_nasdaq_data():
    tickers = []
    name_map = {}
    for item in NASDAQ_CONSTITUENTS:
        sym = item['Symbol']
        if sym != 'CASH': 
            tickers.append(sym)
            name_map[sym] = item['Company']
    return list(set(tickers)), name_map

@st.cache_data(ttl=3600)
def get_top10_tickers():
    sorted_stocks = sorted(NASDAQ_CONSTITUENTS, key=lambda x: x.get('Allocation', 0), reverse=True)
    top10 = [item['Symbol'] for item in sorted_stocks if item['Symbol'] != 'CASH'][:10]
    return top10

# ==========================================
# 2. 數據下載與計算
# ==========================================

@st.cache_data(ttl=3600)
def get_market_data(tickers):
    # 新增下載 HYG (高收益債)
    all_tickers = tickers + ['^NDX', 'TLT', '^VXN', 'HYG'] 
    try:
        data = yf.download(all_tickers, period="2y", group_by='ticker', threads=True, auto_adjust=True)
        return data
    except Exception as e:
        st.error(f"數據下載錯誤: {e}")
        return pd.DataFrame()

def calculate_market_indicators(data, tickers):
    ndx = data['^NDX']['Close']
    tlt = data['TLT']['Close']
    hyg = data['HYG']['Close']
    vxn = data['^VXN']['Close']
    
    benchmark_idx = ndx.index
    valid_tickers = [t for t in tickers if t in data]
    
    close_df = pd.DataFrame({t: data[t]['Close'] for t in valid_tickers}).reindex(benchmark_idx)
    high_df = pd.DataFrame({t: data[t]['High'] for t in valid_tickers}).reindex(benchmark_idx)
    low_df = pd.DataFrame({t: data[t]['Low'] for t in valid_tickers}).reindex(benchmark_idx)
    volume_df = pd.DataFrame({t: data[t]['Volume'] for t in valid_tickers}).reindex(benchmark_idx)
    
    # A. 廣度 (MA60)
    ma60_df = close_df.rolling(window=60).mean()
    above_ma60 = (close_df > ma60_df)
    valid_counts = ma60_df.notna().sum(axis=1)
    above_counts = above_ma60.sum(axis=1)
    breadth_pct = (above_counts / valid_counts * 100).fillna(0)
    
    # B. 累積淨新高
    roll_max_252 = high_df.rolling(window=252).max()
    roll_min_252 = low_df.rolling(window=252).min()
    new_highs = (high_df >= roll_max_252).sum(axis=1)
    new_lows = (low_df <= roll_min_252).sum(axis=1)
    net_nh_nl = new_highs - new_lows
    cum_net_highs = net_nh_nl.cumsum()
    
    # C. 恐慌指標 (VXN)
    vxn_ma50 = vxn.rolling(window=50).mean()
    vxn_ratio = vxn / vxn_ma50
    
    # D. 資產強弱 (NDX vs TLT)
    ndx_ret = ndx.pct_change(20) * 100
    tlt_ret = tlt.pct_change(20) * 100
    strength_diff = ndx_ret - tlt_ret

    # E. TRIN
    daily_change = close_df.diff()
    up_mask = daily_change > 0
    down_mask = daily_change < 0
    advancing_issues = up_mask.sum(axis=1)
    declining_issues = down_mask.sum(axis=1)
    advancing_volume = (volume_df * up_mask).sum(axis=1)
    declining_volume = (volume_df * down_mask).sum(axis=1)
    
    ad_ratio = advancing_issues / declining_issues.replace(0, 1)
    vol_ratio = advancing_volume / declining_volume.replace(0, 1)
    trin = ad_ratio / vol_ratio

    # F. 平均近 20 日上漲-下跌家數 (Net Advances MA20) - 新增
    net_advances = advancing_issues - declining_issues
    ad_ma20 = net_advances.rolling(window=20).mean()

    # G. 風險偏好: 高收益債/美債 (HYG/TLT) - 新增
    # 比值上升代表資金追求高風險收益 (Risk On)，下降代表避險 (Risk Off)
    hyg_tlt_ratio = hyg / tlt
    
    lookback = 130
    return {
        'dates': ndx.index[-lookback:],
        'ndx': ndx.iloc[-lookback:],
        'breadth_pct': breadth_pct.iloc[-lookback:],
        'cum_net_highs': cum_net_highs.iloc[-lookback:], 
        'vxn': vxn.iloc[-lookback:], 
        'vxn_ratio': vxn_ratio.iloc[-lookback:],
        'strength_diff': strength_diff.iloc[-lookback:],
        'trin': trin.iloc[-lookback:],
        'ad_ma20': ad_ma20.iloc[-lookback:], # 新增
        'hyg_tlt': hyg_tlt_ratio.iloc[-lookback:] # 新增
    }

def calculate_top10_rrg(data, top10_tickers):
    ndx = data['^NDX']['Close']
    rrg_data = []
    
    for ticker in top10_tickers:
        if ticker in data:
            close = data[ticker]['Close']
            rs = close / ndx
            rs_trend = rs.rolling(window=10).mean()
            rs_mean = rs_trend.rolling(window=60).mean()
            rs_std = rs_trend.rolling(window=60).std()
            
            x_val = ((rs_trend - rs_mean) / rs_std).iloc[-1]
            x_val_prev = ((rs_trend - rs_mean) / rs_std).iloc[-10]
            y_val = x_val - x_val_prev
            
            df = data[ticker]
            df = df.dropna(subset=['Close'])
            chg = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) >= 2 else 0
            
            rrg_data.append({'Symbol': ticker, 'X': x_val, 'Y': y_val, 'Change': chg})
    return pd.DataFrame(rrg_data)

def get_top10_performance(data, top10_tickers):
    perf = {}
    for ticker in top10_tickers:
        try:
            if ticker in data:
                df = data[ticker]
                df = df.dropna(subset=['Close'])
                if len(df) >= 2:
                    curr = df['Close'].iloc[-1]
                    prev = df['Close'].iloc[-2]
                    change = ((curr - prev) / prev) * 100
                    perf[ticker] = change
        except: continue
    return pd.Series(perf).sort_values(ascending=False)

def get_latest_snapshot_with_strategy(data, tickers, name_map):
    results = []
    for ticker in tickers:
        try:
            if ticker not in data: continue
            df = data[ticker]
            df = df.dropna(subset=['Close', 'Volume'])
            if df.empty or len(df) < 252: continue
            
            curr = df.iloc[-1]
            prev = df.iloc[-2]
            close = float(curr['Close'])
            change_pct = ((close - prev['Close']) / prev['Close']) * 100
            turnover = close * float(curr['Volume'])
            
            ma50 = df['Close'].rolling(50).mean().iloc[-1]
            ma150 = df['Close'].rolling(150).mean().iloc[-1]
            ma200 = df['Close'].rolling(200).mean().iloc[-1]
            high_52w = df['High'].tail(252).max()
            low_52w = df['Low'].tail(252).min()
            
            trend_score = 0
            if close > ma50 > ma150 > ma200: trend_score += 1
            if close > low_52w * 1.3: trend_score += 1
            if close > high_52w * 0.75: trend_score += 1
            is_super_trend = (trend_score == 3)
            
            is_pocket_pivot = False
            if change_pct > 0:
                last_10 = df.iloc[-11:-1]
                down_days = last_10[last_10['Close'] < last_10['Open']]
                if not down_days.empty:
                    if curr['Volume'] > down_days['Volume'].max(): is_pocket_pivot = True
                elif curr['Volume'] > last_10['Volume'].max(): is_pocket_pivot = True

            avg_vol_20 = df['Volume'].iloc[-22:-2].mean()
            r_vol = curr['Volume'] / avg_vol_20 if avg_vol_20 > 0 else 0
            
            ma20 = float(df['Close'].rolling(20).mean().iloc[-1] if len(df) >= 20 else close)
            bias_20 = ((close - ma20) / ma20) * 100
            volatility = ((curr['High'] - curr['Low']) / prev['Close']) * 100

            results.append({
                'Ticker': ticker, 'Name': name_map.get(ticker, ticker),
                'Close': close, 'Change %': change_pct, 'Turnover': turnover,
                'RVol': r_vol, '52W High': high_52w, '52W Low': low_52w,
                'Super Trend': is_super_trend, 'Pocket Pivot': is_pocket_pivot,
                'Bias 20(%)': bias_20, 'Volatility': volatility
            })
        except: continue
    return pd.DataFrame(results)

# ==========================================
# 3. 視覺化
# ==========================================

def main():
    st.title("📊 NASDAQ 100 Pro Market Dashboard")
    st.write(f"Last Update: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    with st.spinner('Downloading & Calculating...'):
        tickers, name_map = parse_nasdaq_data()
        top10_tickers = get_top10_tickers()
        
        full_data = get_market_data(tickers)
        
        if full_data.empty:
            st.error("Data download failed.")
            return

        mkt = calculate_market_indicators(full_data, tickers)
        df_snapshot = get_latest_snapshot_with_strategy(full_data, tickers, name_map)
        rrg_df = calculate_top10_rrg(full_data, top10_tickers)
        top10_perf = get_top10_performance(full_data, top10_tickers)

    x_axis = mkt['dates']

    # 計算 NDX Y軸動態範圍 (Zoom In)
    ndx_min = mkt['ndx'].min()
    ndx_max = mkt['ndx'].max()
    padding = (ndx_max - ndx_min) * 0.05 
    ndx_range = [ndx_min - padding, ndx_max + padding]

    def fmt(df, val_col=None, fmt_str='{:.2f}'):
        d = df.copy()
        d['Close'] = d['Close'].map('{:,.2f}'.format)
        d['Change %'] = d['Change %'].map('{:+.2f}%'.format)
        d['52W High'] = d['52W High'].map('{:,.2f}'.format)
        d['52W Low'] = d['52W Low'].map('{:,.2f}'.format)
        if val_col and val_col in d.columns and fmt_str:
            d[val_col] = d[val_col].map(fmt_str.format)
        return d

    # --- Part 1: 大盤健康度 ---
    st.header("一、 大盤健康度診斷 (Market Health)")
    
    col1_1, col1_2 = st.columns(2)
    with col1_1:
        # Breadth
        fig_breadth = make_subplots(specs=[[{"secondary_y": True}]])
        fig_breadth.add_trace(go.Scatter(x=x_axis, y=mkt['ndx'], name="NDX 100", line=dict(color='black', width=1)), secondary_y=False)
        fig_breadth.add_trace(go.Scatter(x=x_axis, y=mkt['breadth_pct'], name="% > MA60", line=dict(color='blue', width=2), fill='tozeroy', fillcolor='rgba(0,0,255,0.1)'), secondary_y=True)
        fig_breadth.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="50% 分界", secondary_y=True)
        fig_breadth.update_yaxes(range=ndx_range, secondary_y=False) 
        fig_breadth.update_yaxes(title_text="比例 (%)", range=[0, 100], secondary_y=True)
        fig_breadth.update_layout(title="市場廣度：站上 60MA 比例", height=350)
        st.plotly_chart(fig_breadth, use_container_width=True)

    with col1_2:
        # Net Advances
        fig_ad = make_subplots(specs=[[{"secondary_y": True}]])
        fig_ad.add_trace(go.Scatter(x=x_axis, y=mkt['ndx'], name="NDX 100", line=dict(color='black', width=1)), secondary_y=False)
        fig_ad.add_trace(go.Bar(x=x_axis, y=mkt['ad_ma20'], name="20MA Net Advances", marker_color=['green' if v>0 else 'red' for v in mkt['ad_ma20']], opacity=0.6), secondary_y=True)
        fig_ad.add_hline(y=0, line_color="gray", secondary_y=True)
        fig_ad.update_yaxes(range=ndx_range, secondary_y=False)
        fig_ad.update_layout(title="市場動能：平均近 20 日淨上漲家數", height=350)
        st.plotly_chart(fig_ad, use_container_width=True)

    col1_3, col1_4 = st.columns(2)
    with col1_3:
        # Cumul Net Highs
        fig_nhnl = make_subplots(specs=[[{"secondary_y": True}]])
        fig_nhnl.add_trace(go.Scatter(x=x_axis, y=mkt['ndx'], name="NDX 100", showlegend=False, line=dict(color='black', width=1)), secondary_y=False)
        fig_nhnl.add_trace(go.Scatter(x=x_axis, y=mkt['cum_net_highs'], name="Cumul Net Highs", line=dict(color='green', width=2)), secondary_y=True)
        fig_nhnl.update_yaxes(range=ndx_range, secondary_y=False)
        fig_nhnl.update_layout(title="市場趨勢：累積淨新高線 (Cumulative Net Highs)", height=350)
        st.plotly_chart(fig_nhnl, use_container_width=True)

    with col1_4:
        # TRIN
        fig_trin = make_subplots(specs=[[{"secondary_y": True}]])
        fig_trin.add_trace(go.Scatter(x=x_axis, y=mkt['ndx'], name="NDX 100", showlegend=False, line=dict(color='black', width=1)), secondary_y=False)
        fig_trin.add_trace(go.Scatter(x=x_axis, y=mkt['trin'], name="TRIN", line=dict(color='orange', width=2)), secondary_y=True)
        fig_trin.add_hline(y=2.0, line_dash="dot", line_color="red", annotation_text="Panic", secondary_y=True)
        fig_trin.add_hline(y=0.5, line_dash="dot", line_color="green", annotation_text="Greed", secondary_y=True)
        fig_trin.update_yaxes(range=ndx_range, secondary_y=False)
        fig_trin.update_yaxes(range=[0, 3], secondary_y=True)
        fig_trin.update_layout(title="量價結構：TRIN (阿姆斯指數)", height=350)
        st.plotly_chart(fig_trin, use_container_width=True)

    # --- Part 2: 風險控管 ---
    st.header("二、 風險控管 (Risk Management)")
    col2_1, col2_2, col2_3 = st.columns(3)
    
    with col2_1:
        # VXN
        fig_vxn = make_subplots(specs=[[{"secondary_y": True}]])
        fig_vxn.add_trace(go.Scatter(x=x_axis, y=mkt['ndx'], name="NDX 100", showlegend=False, line=dict(color='black', width=1)), secondary_y=False)
        fig_vxn.add_trace(go.Scatter(x=x_axis, y=mkt['vxn_ratio'], name="VXN/MA50 Ratio", line=dict(color='red', width=2)), secondary_y=True)
        fig_vxn.add_hline(y=1.0, line_dash="dot", line_color="gray", annotation_text="Avg Level", secondary_y=True)
        fig_vxn.update_yaxes(range=ndx_range, secondary_y=False)
        fig_vxn.update_layout(title="恐慌結構：VXN 乖離率 (VXN / 50MA)", height=350)
        st.plotly_chart(fig_vxn, use_container_width=True)

    with col2_2:
        # Asset Strength
        fig_asset = make_subplots(specs=[[{"secondary_y": True}]])
        fig_asset.add_trace(go.Scatter(x=x_axis, y=mkt['ndx'], name="NDX 100", showlegend=False, line=dict(color='black', width=1)), secondary_y=False)
        fig_asset.add_trace(go.Scatter(x=x_axis, y=mkt['strength_diff'], name="NDX-TLT Diff", line=dict(color='purple', width=2)), secondary_y=True)
        fig_asset.add_hline(y=0, line_dash="solid", line_color="gray", secondary_y=True)
        fig_asset.update_yaxes(range=ndx_range, secondary_y=False)
        fig_asset.update_layout(title="資產強弱：(NDX - TLT) 20日報酬差值", height=350)
        st.plotly_chart(fig_asset, use_container_width=True)

    with col2_3:
        # Risk Appetite: HYG/TLT
        fig_risk = make_subplots(specs=[[{"secondary_y": True}]])
        fig_risk.add_trace(go.Scatter(x=x_axis, y=mkt['ndx'], name="NDX 100", showlegend=False, line=dict(color='black', width=1)), secondary_y=False)
        fig_risk.add_trace(go.Scatter(x=x_axis, y=mkt['hyg_tlt'], name="HYG/TLT Ratio", line=dict(color='orange', width=2)), secondary_y=True)
        fig_risk.update_yaxes(range=ndx_range, secondary_y=False)
        fig_risk.update_layout(title="風險偏好：高收益債/公債 (HYG / TLT)", height=350)
        st.plotly_chart(fig_risk, use_container_width=True)

    # --- Part 3: 前十大權值股掃描 ---
    st.header("三、 前十大權值股掃描 (Top 10 Holdings)")
    
    # RRG
    fig_rrg = go.Figure()
    fig_rrg.add_trace(go.Scatter(
        x=rrg_df['X'], y=rrg_df['Y'], mode='markers+text', text=rrg_df['Symbol'],
        textposition='top center',
        marker=dict(size=25, color=rrg_df['Change'], colorscale='RdYlGn', showscale=True, colorbar=dict(title="Today %", len=0.5)),
        name="Top 10"
    ))
    fig_rrg.add_vline(x=0, line_width=1, line_dash="dash", line_color="gray")
    fig_rrg.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")
    # 新增象限標籤 (1) RRG Enhancements
    fig_rrg.add_annotation(x=2, y=2, text="領先 (Leading)", showarrow=False, font=dict(size=16, color="green"), opacity=0.5)
    fig_rrg.add_annotation(x=2, y=-2, text="轉弱 (Weakening)", showarrow=False, font=dict(size=16, color="orange"), opacity=0.5)
    fig_rrg.add_annotation(x=-2, y=-2, text="落後 (Lagging)", showarrow=False, font=dict(size=16, color="red"), opacity=0.5)
    fig_rrg.add_annotation(x=-2, y=2, text="改善 (Improving)", showarrow=False, font=dict(size=16, color="blue"), opacity=0.5)
    
    fig_rrg.update_layout(title="前十大權值股 RRG 強弱圖 (右上領先/左下落後)", height=500, xaxis_title="Relative Strength", yaxis_title="Relative Momentum")
    st.plotly_chart(fig_rrg, use_container_width=True)

    # Bar Chart
    fig_perf = go.Figure()
    colors = ['green' if v >= 0 else 'red' for v in top10_perf.values]
    fig_perf.add_trace(go.Bar(
        x=top10_perf.index, y=top10_perf.values, marker_color=colors,
        text=top10_perf.values, texttemplate='%{y:.2f}%', textposition='auto', name="Change"
    ))
    fig_perf.update_layout(title="前十大權值股 今日漲跌幅", height=400)
    st.plotly_chart(fig_perf, use_container_width=True)

    # --- Part 4: 強勢股篩選 ---
    st.header("四、 強勢股篩選 (Stock Selection)")
    
    cols_strat = ['Ticker', 'Name', 'Close', 'Change %', 'RVol', '52W High', '52W Low']
    cols_basic = ['Ticker', 'Name', 'Close', 'Change %', '52W High', '52W Low', 'Val']

    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("🔥 超級趨勢股 (Super Trend)")
        df_super = df_snapshot[df_snapshot['Super Trend'] == True].sort_values('RVol', ascending=False).head(10)
        fig_super = go.Figure(data=[go.Table(
            header=dict(values=cols_strat, fill_color='navy', font=dict(color='white'), align='left'),
            cells=dict(values=[fmt(df_super, 'RVol', '{:.2f}x')[k] for k in cols_strat], fill_color='lavender', align='left'))
        ])
        fig_super.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig_super, use_container_width=True)
        
        st.subheader("🚀 漲幅最強 Top 10")
        gainer_df = df_snapshot.sort_values('Change %', ascending=False).head(10)[['Ticker','Name','Close','Change %','52W High','52W Low','RVol']]
        gainer_df.columns = ['Ticker','Name','Close','Change %','52W High','52W Low','Val']
        fig_gain = go.Figure(data=[go.Table(
            header=dict(values=cols_basic, fill_color='navy', font=dict(color='white'), align='left'),
            cells=dict(values=[fmt(gainer_df, 'Val', '{:.2f}x')[k] for k in cols_basic], fill_color='lavender', align='left'))
        ])
        fig_gain.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig_gain, use_container_width=True)

        st.subheader("⚡ 高波動度 Top 10")
        high_vol = df_snapshot.sort_values('Volatility', ascending=False).head(10)[['Ticker','Name','Close','Change %','52W High','52W Low','Volatility']]
        high_vol.columns = ['Ticker','Name','Close','Change %','52W High','52W Low','Val']
        fig_vol = go.Figure(data=[go.Table(
            header=dict(values=cols_basic, fill_color='navy', font=dict(color='white'), align='left'),
            cells=dict(values=[fmt(high_vol, 'Val', '{:.2f}%')[k] for k in cols_basic], fill_color='lavender', align='left'))
        ])
        fig_vol.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig_vol, use_container_width=True)

    with col4:
        st.subheader("💎 口袋支點爆量 (Pocket Pivot)")
        df_pocket = df_snapshot[df_snapshot['Pocket Pivot'] == True].sort_values('Change %', ascending=False).head(10)
        fig_pocket = go.Figure(data=[go.Table(
            header=dict(values=cols_strat, fill_color='navy', font=dict(color='white'), align='left'),
            cells=dict(values=[fmt(df_pocket, 'RVol', '{:.2f}x')[k] for k in cols_strat], fill_color='lavender', align='left'))
        ])
        fig_pocket.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig_pocket, use_container_width=True)

        st.subheader("💧 跌幅最重 Top 10")
        loser_df = df_snapshot.sort_values('Change %', ascending=True).head(10)[['Ticker','Name','Close','Change %','52W High','52W Low','RVol']]
        loser_df.columns = ['Ticker','Name','Close','Change %','52W High','52W Low','Val']
        fig_loss = go.Figure(data=[go.Table(
            header=dict(values=cols_basic, fill_color='navy', font=dict(color='white'), align='left'),
            cells=dict(values=[fmt(loser_df, 'Val', '{:.2f}x')[k] for k in cols_basic], fill_color='lavender', align='left'))
        ])
        fig_loss.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig_loss, use_container_width=True)

        st.subheader("💥 爆量上漲 Top 10")
        vol_up = df_snapshot[df_snapshot['Change %'] > 0].sort_values('RVol', ascending=False).head(10)[['Ticker','Name','Close','Change %','52W High','52W Low','RVol']]
        vol_up.columns = ['Ticker','Name','Close','Change %','52W High','52W Low','Val']
        fig_volup = go.Figure(data=[go.Table(
            header=dict(values=cols_basic, fill_color='navy', font=dict(color='white'), align='left'),
            cells=dict(values=[fmt(vol_up, 'Val', '{:.2f}x')[k] for k in cols_basic], fill_color='lavender', align='left'))
        ])
        fig_volup.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig_volup, use_container_width=True)

if __name__ == "__main__":
    main()
