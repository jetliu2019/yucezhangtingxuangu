#!/usr/bin/env python3
"""
次日涨停预测选股程序 - 多线程版
多数据源并行采集 → 分片并行筛选 → 并行评分 → PushPlus卡片推送
"""

import akshare as ak
import pandas as pd
import requests
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import wraps

# ===================== 配置区 =====================
PUSHPLUS_TOKEN = "70a87015756f483ab09f70a5ebe5d6ff"  # ← 替换为你的Token

MAX_WORKERS_IO   = 6     # IO线程池（数据采集）
MAX_WORKERS_CPU  = 8     # CPU线程池（筛选评分）
CHUNK_SIZE       = 500   # 每个线程处理的股票数量

PARAMS = {
    "min_change":     5.0,
    "max_change":     9.8,
    "min_turnover":   3.0,
    "max_turnover":  25.0,
    "min_vol_ratio":  1.5,
    "min_amount_wan": 5000,
    "min_price":      3.0,
    "max_price":    100.0,
    "min_tail":      60,
    "top_n":         15,
}
# ==================================================

# ---------- 线程安全计时/日志 ----------
_print_lock = threading.Lock()

def safe_print(msg):
    with _print_lock:
        t = datetime.now().strftime("%H:%M:%S")
        print(f"  [{t}] {msg}")

def timer(func):
    """装饰器：统计函数耗时"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        name = func.__name__
        safe_print(f"⏳ {name} 开始...")
        t0 = time.time()
        result = func(*args, **kwargs)
        cost = time.time() - t0
        safe_print(f"✅ {name} 完成  ({cost:.2f}s)")
        return result
    return wrapper


# =============== 1. 多线程数据采集层 ===============

@timer
def fetch_quotes():
    """获取全A实时行情"""
    df = ak.stock_zh_a_spot_em()
    nums = ['最新价','涨跌幅','成交量','成交额','振幅',
            '最高','最低','今开','昨收','量比','换手率',
            '市盈率-动态','流���市值']
    for c in nums:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    safe_print(f"   → 行情: {len(df)} 只")
    return df


@timer
def fetch_fund_flow():
    """获取个股资金流向"""
    try:
        df = ak.stock_individual_fund_flow_rank(indicator="今日")
        main_col = [c for c in df.columns if '主力净流入' in c and '净额' in c]
        if main_col:
            df['主力净流入'] = pd.to_numeric(df[main_col[0]], errors='coerce')
        else:
            df['主力净流入'] = 0
        safe_print(f"   → 资金流向: {len(df)} 条")
        return df[['代码', '主力净流入']].copy()
    except Exception as e:
        safe_print(f"   ⚠️  资金流向失败: {e}")
        return pd.DataFrame(columns=['代码', '主力净流入'])


@timer
def fetch_limit_up():
    """获取今日涨停/曾涨停股票（情绪参考）"""
    try:
        df = ak.stock_zt_pool_em(date=datetime.now().strftime('%Y%m%d'))
        codes = set(df['代码'].tolist()) if '代码' in df.columns else set()
        safe_print(f"   → 涨停池: {len(codes)} 只")
        return codes
    except Exception as e:
        safe_print(f"   ⚠️  涨停池获取失败: {e}")
        return set()


@timer
def fetch_strong_pool():
    """获取强势股池（昨日涨停今日未跌停）"""
    try:
        df = ak.stock_zt_pool_strong_em(date=datetime.now().strftime('%Y%m%d'))
        codes = set(df['代码'].tolist()) if '代码' in df.columns else set()
        safe_print(f"   → 强势池: {len(codes)} 只")
        return codes
    except Exception as e:
        safe_print(f"   ⚠️  强势池获取失败: {e}")
        return set()


@timer
def fetch_sector_flow():
    """获取板块资金流向 TOP（标记热门板块）"""
    try:
        df = ak.stock_sector_fund_flow_rank(
            indicator="今日", sector_type="行业资金流"
        )
        if not df.empty:
            top5 = df.head(5)['名称'].tolist()
            safe_print(f"   → 热门板块: {', '.join(top5)}")
            return top5
        return []
    except Exception as e:
        safe_print(f"   ⚠️  板块资金失败: {e}")
        return []


def parallel_fetch_all():
    """
    并行采集所有数据源（核心加速点）
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    串行: ~25s   →   并行: ~6s (提速4x+)
    """
    safe_print("🚀 并行采集 6 个数据源...")
    t0 = time.time()

    results = {}
    tasks = {
        "quotes":      fetch_quotes,
        "fund_flow":   fetch_fund_flow,
        "limit_up":    fetch_limit_up,
        "strong_pool": fetch_strong_pool,
        "sector_flow": fetch_sector_flow,
    }

    with ThreadPoolExecutor(max_workers=MAX_WORKERS_IO, thread_name_prefix="IO") as pool:
        future_map = {pool.submit(fn): name for name, fn in tasks.items()}
        for future in as_completed(future_map):
            name = future_map[future]
            try:
                results[name] = future.result()
            except Exception as e:
                safe_print(f"❌ {name} 异常: {e}")
                results[name] = None

    cost = time.time() - t0
    safe_print(f"📦 全部数据采集完成  总耗时 {cost:.2f}s\n")
    return results


# =============== 2. 多线程筛选评分层 ===============

def basic_filter(chunk, params):
    """基础条件过滤（线程安全，无共享状态）"""
    df = chunk.copy()
    p = params

    # 排除 ST / 退市
    df = df[~df['名称'].str.contains('ST|st|退', na=False)]
    # 仅保留 主板 + 创业板
    df = df[df['代码'].str.match('^(00|30|60)')]

    # 数值筛选
    cond = (
        df['涨跌幅'].between(p['min_change'], p['max_change']) &
        df['换手率'].between(p['min_turnover'], p['max_turnover']) &
        (df['量比']  >= p['min_vol_ratio']) &
        df['最新价'].between(p['min_price'], p['max_price']) &
        (df['成交额'] >= p['min_amount_wan'] * 1e4)
    )
    df = df[cond]

    # 尾盘强度
    spread = df['最高'] - df['最低']
    df = df[spread > 0].copy()
    df['尾盘强度'] = ((df['最新价'] - df['最低']) / spread * 100).round(1)
    df = df[df['尾盘强度'] >= p['min_tail']]

    return df


def calc_score(row, limit_up_set, strong_set):
    """综合评分（增加涨停/强势加分）"""
    s = 0
    # ① 涨幅（35分）
    s += min(row['涨跌幅'] / 10 * 35, 35)
    # ② 尾盘强度（25分）
    s += row['尾盘强度'] / 100 * 25
    # ③ 量比（15分）
    s += min(row['量比'] / 4 * 15, 15)
    # ④ 换手率（15分）
    tr = row['换手率']
    if 5 <= tr <= 15:
        s += 15
    elif 3 <= tr <= 25:
        s += 8
    # ⑤ 主力净流入（10分）
    mi = row.get('主力净流入', 0)
    if mi > 5000:   s += 10
    elif mi > 1000: s += 6
    elif mi > 0:    s += 3

    # 🔥 额外加分
    code = row['代码']
    if code in strong_set:          # 强势股连板潜力
        s += 5
    if code in limit_up_set:        # 曾触及涨停
        s += 3

    return round(min(s, 100), 1)


def process_chunk(chunk_id, chunk, params, flows, limit_up_set, strong_set):
    """处理单个分片：过滤 → 合并资金 → 评分"""
    # 基础过滤
    filtered = basic_filter(chunk, params)
    if filtered.empty:
        return filtered

    # 合并资金流向
    if not flows.empty:
        filtered = filtered.merge(flows, on='代码', how='left')
        filtered['主力净流入'] = filtered['主力净流入'].fillna(0)
        filtered = filtered[filtered['主力净流入'] > 0]
    else:
        filtered['主力净流入'] = 0

    if filtered.empty:
        return filtered

    # 评分
    filtered['评分'] = filtered.apply(
        lambda r: calc_score(r, limit_up_set, strong_set), axis=1
    )

    safe_print(f"   分片#{chunk_id}: {len(chunk)}只 → 筛选出 {len(filtered)}只")
    return filtered


def parallel_screen(quotes, flows, limit_up_set, strong_set):
    """
    分片并行筛选 + 评分
    ━━━━━━━━━━━━━━━━━━━━━
    5000只 / 500片 = 10个任务 → 8线程并行
    """
    safe_print("🔍 多线程分片筛选...")
    t0 = time.time()

    # 切分数据
    chunks = [quotes.iloc[i:i+CHUNK_SIZE] for i in range(0, len(quotes), CHUNK_SIZE)]
    safe_print(f"   共 {len(quotes)} 只 → 分成 {len(chunks)} 个分片 (每片{CHUNK_SIZE}只)")

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_CPU, thread_name_prefix="CPU") as pool:
        futures = {
            pool.submit(
                process_chunk, i, chunk, PARAMS, flows, limit_up_set, strong_set
            ): i
            for i, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            try:
                df = future.result()
                if df is not None and not df.empty:
                    results.append(df)
            except Exception as e:
                cid = futures[future]
                safe_print(f"   ❌ 分片#{cid} 异常: {e}")

    # 合并所有分片结果
    if results:
        final = pd.concat(results, ignore_index=True)
        final = final.sort_values('评分', ascending=False).head(PARAMS['top_n']).reset_index(drop=True)
    else:
        final = pd.DataFrame()

    cost = time.time() - t0
    safe_print(f"🏁 筛选完成: {len(final)} 只候选  ({cost:.2f}s)\n")
    return final


# =============== 3. 输出与推送层 ===============

def fmt_amount(val):
    if abs(val) >= 1e8:
        return f"{val/1e8:.2f}亿"
    return f"{val/1e4:.0f}万"


def console_print(df):
    if df.empty:
        print("\n   📭 无符合条件的股票\n")
        return
    print(f"\n{'━'*84}")
    print(f" {'#':>2}  {'代码':>8}  {'名称':<6}  {'涨幅':>7}  {'换手':>7}  "
          f"{'量比':>5}  {'尾盘':>5}  {'主力净流入':>10}  {'评分':>5}")
    print(f"{'━'*84}")
    for i, r in df.iterrows():
        mi_str = fmt_amount(r['主力净流入'])
        print(f" {i+1:>2}  {r['代码']:>8}  {r['名称']:<6}  "
              f"{r['涨跌幅']:>+6.2f}%  {r['换手率']:>6.2f}%  "
              f"{r['量比']:>5.2f}  {r['尾盘强度']:>4.0f}%  "
              f"{mi_str:>10}  {r['评分']:>5.1f}")
    print(f"{'━'*84}\n")


def build_html(df, hot_sectors, stats):
    """生成精美卡片 HTML（含统计和板块信息）"""
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    p = PARAMS

    # 头部统计栏
    sector_str = '、'.join(hot_sectors[:5]) if hot_sectors else '暂无'
    header_stats = f"""
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;
                margin:10px 0;text-align:center;">
      <div style="background:#f0f9ff;border-radius:8px;padding:8px;">
        <div style="font-size:11px;color:#888;">数据采集</div>
        <div style="font-size:16px;font-weight:bold;color:#3498db;">{stats['fetch_time']:.1f}s</div>
      </div>
      <div style="background:#fef9e7;border-radius:8px;padding:8px;">
        <div style="font-size:11px;color:#888;">分析筛选</div>
        <div style="font-size:16px;font-weight:bold;color:#f39c12;">{stats['screen_time']:.1f}s</div>
      </div>
      <div style="background:#f0fff0;border-radius:8px;padding:8px;">
        <div style="font-size:11px;color:#888;">候选数量</div>
        <div style="font-size:16px;font-weight:bold;color:#27ae60;">{len(df)}只</div>
      </div>
    </div>
    <div style="background:#f8f9fa;border-radius:6px;padding:8px 12px;
                font-size:11px;color:#666;margin-bottom:8px;">
      🔥 今日热门板块：<b>{sector_str}</b>
    </div>"""

    if df.empty:
        return f"""<div style="font-family:sans-serif;max-width:480px;margin:0 auto;padding:10px;">
            <h2 style="text-align:center;color:#e74c3c;">🔥 次日涨停预测</h2>
            {header_stats}
            <h3 style="text-align:center;color:#999;">📭 今日无符合条件的股票</h3></div>"""

    cards = ""
    for i, r in df.iterrows():
        mi = r['主力净流入']
        mi_str = fmt_amount(mi)
        mi_color = "#e74c3c" if mi > 0 else "#27ae60"
        amt_str = fmt_amount(r['成交额'])
        mcap = r.get('流通市值', 0)
        mcap_str = fmt_amount(mcap) if mcap and mcap > 0 else "-"

        # 评分颜色
        sc = r['评分']
        sc_color = "#e74c3c" if sc >= 80 else "#f39c12" if sc >= 60 else "#3498db"

        # 评分条
        bar_width = min(sc, 100)

        cards += f"""
        <div style="border:1px solid #f0f0f0;border-left:4px solid {sc_color};
                    border-radius:10px;padding:14px 16px;margin:10px 0;
                    background:linear-gradient(135deg,#fff 0%,#fefefe 100%);
                    box-shadow:0 2px 8px rgba(0,0,0,0.06);">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <div>
              <span style="background:{sc_color};color:#fff;border-radius:4px;
                           padding:2px 7px;font-size:11px;margin-right:6px;">No.{i+1}</span>
              <b style="font-size:17px;">{r['名称']}</b>
              <span style="color:#aaa;font-size:12px;margin-left:6px;">{r['代码']}</span>
            </div>
            <div style="text-align:right;">
              <div style="font-size:22px;font-weight:bold;color:#e74c3c;">+{r['涨跌幅']:.2f}%</div>
            </div>
          </div>
          <div style="margin:8px 0 4px;background:#f0f0f0;border-radius:4px;height:6px;">
            <div style="width:{bar_width}%;height:100%;border-radius:4px;
                        background:linear-gradient(90deg,{sc_color},#f39c12);"></div>
          </div>
          <div style="text-align:right;font-size:11px;color:{sc_color};font-weight:bold;">
            ⭐ {sc}分
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;
                      margin-top:6px;font-size:12px;color:#888;">
            <div>现价 <b style="color:#333;">{r['最新价']:.2f}</b></div>
            <div>换手 <b style="color:#333;">{r['换手率']:.2f}%</b></div>
            <div>量比 <b style="color:#333;">{r['量比']:.2f}</b></div>
            <div>振幅 <b style="color:#333;">{r['振幅']:.2f}%</b></div>
            <div>尾盘 <b style="color:#333;">{r['尾盘强度']:.0f}%</b></div>
            <div>成交 <b style="color:#333;">{amt_str}</b></div>
          </div>
          <div style="display:flex;justify-content:space-between;margin-top:8px;
                      font-size:12px;color:#888;">
            <div>💰 主力净流入 <b style="color:{mi_color};">{mi_str}</b></div>
            <div>🏷️ 流通市值 <b style="color:#333;">{mcap_str}</b></div>
          </div>
        </div>"""

    html = f"""
    <div style="font-family:-apple-system,Helvetica,Arial,sans-serif;
                max-width:480px;margin:0 auto;padding:10px;">
      <h2 style="text-align:center;color:#e74c3c;margin-bottom:2px;">
        🔥 次日涨停预测选股
      </h2>
      <p style="text-align:center;color:#aaa;font-size:10px;margin:4px 0 6px;">
        {now} &nbsp;|&nbsp; 多线程采集 · {stats['total_stocks']}只扫描
        · {len(df)}只入选
      </p>
      {header_stats}
      <div style="padding:6px 12px;background:#fff8e1;
                  border-radius:6px;font-size:11px;color:#f39c12;text-align:center;
                  margin-bottom:4px;">
        📌 筛选条件: 涨幅{p['min_change']}~{p['max_change']}%
        · 换手≥{p['min_turnover']}% · 量比≥{p['min_vol_ratio']}
        · 尾盘≥{p['min_tail']}% · 主力净流入>0
      </div>
      {cards}
      <p style="text-align:center;color:#ccc;font-size:10px;margin-top:16px;">
        ⚠️ 以上数据仅供参考，不构成任何投资建议<br>
        请结合板块/题材/大盘综合判断
      </p>
    </div>"""
    return html


def pushplus_send(title, content):
    safe_print("📱 推送到手机...")
    try:
        resp = requests.post(
            "http://www.pushplus.plus/send",
            json={
                "token":    PUSHPLUS_TOKEN,
                "title":    title,
                "content":  content,
                "template": "html",
            },
            timeout=30,
        )
        res = resp.json()
        if res.get("code") == 200:
            safe_print("✅ 推送成功！请查看手机")
        else:
            safe_print(f"❌ 推送失败: {res.get('msg')}")
    except Exception as e:
        safe_print(f"❌ 推送异常: {e}")


# ===================== 主入口 =====================

def main():
    banner = """
    ╔═══════════════════════════════════════════╗
    ║    🚀  涨停预测选股 · 多线程版  v2.0      ║
    ║    IO线程: {:<2d}  |  CPU线程: {:<2d}  |  分片: {:<4d}║
    ╚═══════════════════════════════════════════╝
    """.format(MAX_WORKERS_IO, MAX_WORKERS_CPU, CHUNK_SIZE)
    print(banner)

    total_t0 = time.time()

    # ━━━━━━━━━━ 阶段1: 并行数据采集 ━━━━━━━━━━
    fetch_t0 = time.time()
    data = parallel_fetch_all()
    fetch_time = time.time() - fetch_t0

    quotes      = data.get("quotes")
    flows       = data.get("fund_flow",   pd.DataFrame())
    limit_up    = data.get("limit_up",    set())
    strong_pool = data.get("strong_pool", set())
    hot_sectors = data.get("sector_flow", [])

    if quotes is None or quotes.empty:
        safe_print("❌ 行情数据为空，退出")
        return

    # ━━━━━━━━━━ 阶段2: 多线程筛选评分 ━━━━━━━━━━
    screen_t0 = time.time()
    if flows is None:
        flows = pd.DataFrame(columns=['代码', '主力净流入'])
    if limit_up is None:
        limit_up = set()
    if strong_pool is None:
        strong_pool = set()

    result = parallel_screen(quotes, flows, limit_up, strong_pool)
    screen_time = time.time() - screen_t0

    # ━━━━━━━━━━ 阶段3: 输出与推送 ━━━━━━━━━━
    console_print(result)

    stats = {
        "fetch_time":   fetch_time,
        "screen_time":  screen_time,
        "total_stocks": len(quotes),
    }

    html = build_html(result, hot_sectors or [], stats)
    today = datetime.now().strftime("%m-%d")
    title = f"🔥 涨停预测  {len(result)}只候选-[{today}]"
    pushplus_send(title, html)

    total_cost = time.time() - total_t0
    print(f"""
    ╔═══════════════════════════════════════════╗
    ║              ⏱️  耗时统计                  ║
    ║  数据采集:  {fetch_time:>6.2f}s  (并行{MAX_WORKERS_IO}线程)     ║
    ║  筛选评分:  {screen_time:>6.2f}s  (并行{MAX_WORKERS_CPU}线程)     ║
    ║  总耗时:    {total_cost:>6.2f}s                     ║
    ╚═══════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
