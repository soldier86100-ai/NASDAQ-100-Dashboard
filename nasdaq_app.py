import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import numpy as np

# 設定網頁寬度與標題
st.set_page_config(layout="wide", page_title="NASDAQ 100 Market Dashboard")

# ==========================================
# 1. 核心數據定義 (NASDAQ 100 Constituents)
# ==========================================

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
    # 提取所有 Symbol，過濾掉 CASH
    tickers = [item['Symbol'] for item in NASDAQ_CONSTITUENTS if item['Symbol'] != 'CASH']
    # 建立一個簡單的 Company Name 對照表 (Option)
    name_map = {item['Symbol']: item['Company'] for item in NASDAQ_CONSTITUENTS}
    return list(set(tickers)), name_map

# ==========================================
# 2. 數據下載與計算
# ==========================================

@st.cache_data(ttl=3600)
def get_market_data(tickers):
    # ^NDX = Nasdaq 100 Index
    # ^VXN = Cboe NASDAQ-100 Volatility Index (那斯達克版的恐慌指數)
    # TLT = 20年美債 (通用)
    all_tickers = tickers + ['^NDX', 'TLT', '^VXN'] 
    try:
        # 下載 2 年歷史數據
        data = yf.download(all_tickers, period="2y", group_by='ticker', threads=True, auto_adjust=True)
        return data
    except Exception as e:
        st.error(f"數據下載錯誤: {e}")
        return pd.DataFrame()

def calculate_market_indicators(data, tickers):
    # 1. 準備基準數據
    ndx = data['^NDX']['Close']
    tlt = data['TLT']['Close']
    vxn = data['^VXN']['Close']
    benchmark_idx = ndx.index
    
    # 2. 建立個股矩陣
    valid_tickers = [t for t in tickers if t in data]
    close_df = pd.DataFrame({t: data[t]['Close'] for t in valid_tickers}).reindex(benchmark_idx)
    high_df = pd.DataFrame({t: data[t]['High'] for t in valid_tickers}).reindex(benchmark_idx)
    low_df = pd.DataFrame({t: data[t]['Low'] for t in valid_tickers}).reindex(benchmark_idx)
    
    # A. 市場廣度 (MA60)
    ma60_df = close_df.rolling(window=60).mean()
    above_ma60 = (close_df > ma60_df)
    valid_counts = ma60_df.notna().sum(axis=1)
    above_counts = above_ma60.sum(axis=1)
    breadth_pct = (above_counts / valid_counts * 100).fillna(0)
    
    # B. 52 週新高/新低比率
    roll_max_252 = high_df.rolling(window=252).max()
    roll_min_252 = low_df.rolling(window=252).min()
    new_highs = (high_df >= roll_max_252).sum(axis=1)
    new_lows = (low_df <= roll_min_252).sum(axis=1)
    
    safe_lows = new_lows.replace(0, 1) 
    nh_nl_ratio = new_highs / safe_lows
    
    # C. 騰落指標 (A/D Line 20MA)
    daily_change = close_df.diff()
    advancing = (daily_change > 0).sum(axis=1)
    declining = (daily_change < 0).sum(axis=1)
    net_adv_dec = advancing - declining
    ad_ma20 = net_adv_dec.rolling(window=20).mean()
    
    # D. 資產強弱 (NDX 報酬 - TLT 報酬)
    ndx_ret_20 = ndx.pct_change(20) * 100
    tlt_ret_20 = tlt.pct_change(20) * 100
    strength_diff = ndx_ret_20 - tlt_ret_20
    
    # E. VXN 均線
    vxn_ma50 = vxn.rolling(window=50).mean()

    lookback = 130
    return {
        'dates': ndx.index[-lookback:],
        'ndx': ndx.iloc[-lookback:],
        'breadth_pct': breadth_pct.iloc[-lookback:],
        'nh_nl_ratio': nh_nl_ratio.iloc[-lookback:], 
        'ad_ma20': ad_ma20.iloc[-lookback:],
        'strength_diff': strength_diff.iloc[-lookback:],
        'vxn': vxn.iloc[-lookback:],
        'vxn_ma50': vxn_ma50.iloc[-lookback:]
    }

def get_top_holdings_performance(data):
    """計算前 10 大權重股的今日漲跌幅"""
    # 根據 Allocation 排序取前 10
    top_10 = sorted(NASDAQ_CONSTITUENTS, key=lambda x: x.get('Allocation', 0), reverse=True)[:10]
    
    performance = {}
    
    for item in top_10:
        ticker = item['Symbol']
        try:
            if ticker in data:
                df = data[ticker]
                df = df.dropna(subset=['Close'])
                if len(df) >= 2:
                    curr = df['Close'].iloc[-1]
                    prev = df['Close'].iloc[-2]
                    change = ((curr - prev) / prev) * 100
                    # 使用 Symbol 作為 key，但我們會顯示公司名稱或代碼
                    performance[ticker] = change
        except:
            continue
            
    # 回傳 Series
    return pd.Series(performance).sort_values(ascending=False)

def get_latest_snapshot(data, tickers, name_map):
    results = []
    for ticker in tickers:
        try:
            if ticker not in data: continue
            df = data[ticker]
            df = df.dropna(subset=['Close', 'Volume'])
            if df.empty or len(df) < 252: continue 
            
            curr = df.iloc[-1]
            prev = df.iloc[-2]
            
            change_pct = ((curr['Close'] - prev['Close']) / prev['Close']) * 100
            turnover = curr['Close'] * curr['Volume']
            ma20 = float(df['Close'].rolling(20).mean().iloc[-1])
            bias_20 = ((curr['Close'] - ma20) / ma20) * 100
            volatility = ((curr['High'] - curr['Low']) / prev['Close']) * 100
            avg_vol_20 = df['Volume'].iloc[-22:-2].mean()
            r_vol = curr['Volume'] / avg_vol_20 if avg_vol_20 > 0 else 0
            
            high_52w = df['High'].tail(252).max()
            low_52w = df['Low'].tail(252).min()
            
            results.append({
                'Ticker': ticker,
                'Name': name_map.get(ticker, ticker), # 加入公司名稱
                'Close': curr['Close'],
                'Change %': change_pct,
                'Turnover': turnover,
                'Bias 20(%)': bias_20,
                'Volatility': volatility,
                'RVol': r_vol,
                '52W High': high_52w,
                '52W Low': low_52w
            })
        except:
            continue
    return pd.DataFrame(results)

# ==========================================
# 3. 視覺化與 Streamlit 呈現
# ==========================================

def main():
    st.title("📊 NASDAQ 100 Advanced Market Dashboard")
    st.write(f"Last Update: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.info("首次載入可能需要 30-60 秒下載完整成分股數據，請稍候...")
    
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    with st.spinner('Downloading NASDAQ 100 Data...'):
        tickers, name_map = parse_nasdaq_data()
        full_data = get_market_data(tickers)
    
    if full_data.empty:
        st.error("Failed to download data.")
        return

    with st.spinner('Calculating Indicators...'):
        mkt = calculate_market_indicators(full_data, tickers)
        df_snapshot = get_latest_snapshot(full_data, tickers, name_map)
        
        # 計算前十大權值股表現
        top_perf = get_top_holdings_performance(full_data)

    # 繪圖 Layout
    fig = make_subplots(
        rows=10, cols=2,
        column_widths=[0.5, 0.5],
        row_heights=[0.1, 0.1, 0.1, 0.1, 0.1, 0.15, 0.08, 0.08, 0.08, 0.08],
        specs=[
            [{"colspan": 2, "secondary_y": True}, None], # R1
            [{"colspan": 2, "secondary_y": True}, None], # R2
            [{"colspan": 2, "secondary_y": True}, None], # R3
            [{"colspan": 2, "secondary_y": True}, None], # R4
            [{"colspan": 2, "secondary_y": True}, None], # R5
            [{"colspan": 2, "secondary_y": False}, None],# R6: Top Holdings Perf
            [{"type": "table"}, {"type": "table"}],
            [{"type": "table"}, {"type": "table"}],
            [{"type": "table"}, {"type": "table"}],
            [{"type": "table"}, {"type": "table"}]
        ],
        vertical_spacing=0.06,
        subplot_titles=(
            "市場廣度：站上 60MA 比例 vs NASDAQ 100",
            "市場內部：52週新高/新低 家數比率 (Highs/Lows Ratio) - Nasdaq 100",
            "市場動能：20日平均淨上漲家數 (Net Adv-Dec) vs NASDAQ 100",
            "資產強弱：(NDX 20日報酬 - TLT 20日報酬) 差值 (折線圖)",
            "恐慌指數：VXN (Nasdaq VIX) vs 50日均線",
            "前 10 大權值股今日漲跌幅 (Top 10 Holdings Performance)",
            "1. 漲幅最強 10 檔", "2. 跌幅最重 10 檔",
            "3. 高波動度", "6. 正乖離過大 (>MA20)",
            "7. 負乖離過大 (<MA20)", "4. 爆量上漲",
            "5. 爆量下跌", ""
        )
    )

    x_axis = mkt['dates']

    # R1: Breadth
    fig.add_trace(go.Scatter(x=x_axis, y=mkt['ndx'], name="NDX", line=dict(color='black', width=1)), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=x_axis, y=mkt['breadth_pct'], name="% > MA60", line=dict(color='blue', width=2), fill='tozeroy', fillcolor='rgba(0,0,255,0.1)'), row=1, col=1, secondary_y=True)

    # R2: NH/NL Ratio (Line Chart)
    fig.add_trace(go.Scatter(x=x_axis, y=mkt['ndx'], name="NDX", showlegend=False, line=dict(color='black', width=1)), row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=x_axis, y=mkt['nh_nl_ratio'], name="Highs/Lows Ratio", line=dict(color='green', width=2)), row=2, col=1, secondary_y=True)
    fig.add_hline(y=1, line_dash="dash", line_color="gray", row=2, col=1, secondary_y=True)

    # R3: A/D Line
    ad_colors = ['green' if v >= 0 else 'red' for v in mkt['ad_ma20']]
    fig.add_trace(go.Scatter(x=x_axis, y=mkt['ndx'], name="NDX", showlegend=False, line=dict(color='black', width=1)), row=3, col=1, secondary_y=False)
    fig.add_trace(go.Bar(x=x_axis, y=mkt['ad_ma20'], name="20MA Net Adv-Dec", marker_color=ad_colors, opacity=0.6), row=3, col=1, secondary_y=True)

    # R4: Asset Strength (Line Chart)
    fig.add_trace(go.Scatter(x=x_axis, y=mkt['ndx'], name="NDX", showlegend=False, line=dict(color='black', width=1)), row=4, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=x_axis, y=mkt['strength_diff'], name="NDX - TLT Return Diff", line=dict(color='purple', width=2)), row=4, col=1, secondary_y=True)
    fig.add_hline(y=0, line_dash="solid", line_color="gray", row=4, col=1, secondary_y=True)

    # R5: VXN
    fig.add_trace(go.Scatter(x=x_axis, y=mkt['ndx'], name="NDX", showlegend=False, line=dict(color='black', width=1)), row=5, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=x_axis, y=mkt['vxn'], name="VXN", line=dict(color='red', width=1)), row=5, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=x_axis, y=mkt['vxn_ma50'], name="VXN MA50", line=dict(color='darkred', width=1.5, dash='dash')), row=5, col=1, secondary_y=True)

    # R6: Top 10 Holdings Performance
    sect_colors = ['green' if v >= 0 else 'red' for v in top_perf.values]
    fig.add_trace(go.Bar(
        x=top_perf.index, 
        y=top_perf.values,
        marker_color=sect_colors,
        text=top_perf.values,
        texttemplate='%{y:.2f}%',
        textposition='auto',
        name="Top 10 Change"
    ), row=6, col=1)

    # Tables
    def add_table(row, col, df, cols=['Ticker', 'Close', 'Chg%', '52W High', '52W Low', 'Val']):
        fig.add_trace(go.Table(
            header=dict(values=cols, fill_color='navy', font=dict(color='white'), align='left'),
            cells=dict(values=[df[k] for k in df.columns], fill_color='lavender', align='left')
        ), row=row, col=col)

    def fmt(df, val_col, format_str):
        d = df[['Ticker', 'Close', 'Change %', '52W High', '52W Low', val_col]].copy()
        d['Close'] = d['Close'].map('{:,.2f}'.format)
        d['Change %'] = d['Change %'].map('{:+.2f}%'.format)
        d['52W High'] = d['52W High'].map('{:,.2f}'.format)
        d['52W Low'] = d['52W Low'].map('{:,.2f}'.format)
        d[val_col] = d[val_col].map(format_str.format)
        return d

    add_table(7, 1, fmt(df_snapshot.sort_values('Change %', ascending=False).head(10), 'RVol', '{:.2f}x'))
    add_table(7, 2, fmt(df_snapshot.sort_values('Change %', ascending=True).head(10), 'RVol', '{:.2f}x'))
    add_table(8, 1, fmt(df_snapshot.sort_values('Volatility', ascending=False).head(10), 'Volatility', '{:.2f}%'))
    add_table(8, 2, fmt(df_snapshot.sort_values('Bias 20(%)', ascending=False).head(10), 'Bias 20(%)', '{:+.2f}%'))
    add_table(9, 1, fmt(df_snapshot.sort_values('Bias 20(%)', ascending=True).head(10), 'Bias 20(%)', '{:+.2f}%'))
    vol_up = df_snapshot[df_snapshot['Change %'] > 0].sort_values('RVol', ascending=False).head(10)
    add_table(9, 2, fmt(vol_up, 'RVol', '{:.2f}x'))
    vol_down = df_snapshot[df_snapshot['Change %'] < 0].sort_values('RVol', ascending=False).head(10)
    add_table(10, 1, fmt(vol_down, 'RVol', '{:.2f}x'))

    fig.update_layout(height=3000, template="plotly_white", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
