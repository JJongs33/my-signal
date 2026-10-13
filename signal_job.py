"""Weekly job for the '쫑우의 투자신호' app (runs on GitHub Actions every Saturday 08:00 KST).

1. Computes the rates + dxy score exactly as in the backtest:
     rates = clip( -(10y Treasury yield [1-day lag] - its 126-trading-day mean) / 0.5 , -1, 1 )
     dxy   = clip( -(US dollar index / its 100-trading-day mean - 1) / 0.03 , -1, 1 )
     score = (rates + dxy) / 2      on the NASDAQ-100 trading calendar, read at the last trading day of each week
2. Writes docs/data.json for the app.
3. Sends a web-push notification to every subscribed phone (secrets VAPID_PRIVATE_KEY and PUSH_SUBSCRIPTIONS).

    python signal_job.py            # compute, write data, push (if secrets are set)
    python signal_job.py --no-push  # compute and write only
"""
import argparse
import io
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import yfinance as yf

THRESHOLD = -0.30  # score at or below: sell 5% of the holding this week
BUY_LEVEL = 0.0    # score at or above: buy back with 25% of the cash from earlier sales
SELL_PCT, BUY_PCT = 5, 25
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "docs", "data.json")
NAMES = {"SELL": "매도", "BUY": "재매수", "HOLD": "관망"}


def fred(series_id):
    r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}", timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["date", "value"]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.set_index(pd.to_datetime(df["date"]))["value"].dropna()


def close(ticker, start="2012-01-01"):
    h = yf.Ticker(ticker).history(start=start, auto_adjust=False)
    s = h["Close"].dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s[~s.index.duplicated(keep="last")]


def state_of(score):
    return "SELL" if score <= THRESHOLD else ("BUY" if score >= BUY_LEVEL else "HOLD")


def compute():
    ndx = close("^NDX")
    full = ndx.index
    y = fred("DGS10").reindex(full, method="ffill").shift(1)
    dxy = close("DX-Y.NYB").reindex(full, method="ffill")
    qld = close("QLD").reindex(full, method="ffill")
    sc = pd.DataFrame(index=full)
    sc["rates"] = (-(y - y.rolling(126).mean()) / 0.5).clip(-1, 1)
    sc["dxy"] = (-(dxy / dxy.rolling(100).mean() - 1) / 0.03).clip(-1, 1)
    sc["score"] = (sc["rates"] + sc["dxy"]) / 2
    sc["y10"], sc["y10_mean"] = y, y.rolling(126).mean()
    sc["dxy_lvl"], sc["dxy_mean"] = dxy, dxy.rolling(100).mean()
    sc["ndx"], sc["qld"] = ndx, qld
    iso = sc.index.isocalendar()
    wk = sc.groupby([iso.year.values, iso.week.values]).tail(1).dropna(subset=["score"])
    return wk


def build(wk):
    wk = wk.copy()
    wk["state"] = [state_of(v) for v in wk["score"]]
    row, prev = wk.iloc[-1], wk.iloc[-2]
    run = 0
    for s in wk["state"].to_numpy()[::-1]:
        if s == row["state"]:
            run += 1
        else:
            break
    hist = wk.iloc[-157:]
    ndx0, qld0 = hist["ndx"].iloc[0], hist["qld"].iloc[0]
    price_hist = [{"d": str(d.date()), "n": round(float(v / ndx0 * 100), 2), "q": round(float(w / qld0 * 100), 2)}
                  for d, v, w in zip(hist.index, hist["ndx"], hist["qld"])]
    changes = []
    st = wk["state"].to_numpy()
    for i in range(1, len(wk)):
        if st[i] != st[i - 1] and wk.index[i] >= wk.index[-1] - pd.Timedelta(days=3 * 365):
            changes.append({"d": str(wk.index[i].date()), "from": NAMES[st[i - 1]], "to": NAMES[st[i]], "s": round(float(wk["score"].iloc[i]), 2)})
    r = lambda x: round(float(x), 3)
    data = {
        "updated": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        "asof": str(row.name.date()), "state": row["state"], "state_kr": NAMES[row["state"]], "run": run,
        "score": r(row["score"]), "rates": r(row["rates"]), "dxy": r(row["dxy"]),
        "y10": r(row["y10"]), "y10_mean": r(row["y10_mean"]), "dxy_lvl": r(row["dxy_lvl"]), "dxy_mean": r(row["dxy_mean"]),
        "prev_state_kr": NAMES[prev["state"]], "prev_score": r(prev["score"]),
        "threshold": THRESHOLD, "buy_level": BUY_LEVEL, "sell_pct": SELL_PCT, "buy_pct": BUY_PCT,
        "history": [{"d": str(d.date()), "s": round(float(v), 3)} for d, v in hist["score"].items()],
        "price_history": price_hist,
        "ndx_now": round(float(row["ndx"]), 1), "qld_now": round(float(row["qld"]), 2),
        "changes": changes[::-1],
    }
    return data


def message(d):
    if d["state"] == "SELL":
        head = f"매도 {d['run']}주째: 다음 거래일에 보유분의 {d['sell_pct']}% 매도 -> 현금 보관"
    elif d["state"] == "BUY":
        head = f"재매수: 매도로 쌓인 현금이 있으면 그 {d['buy_pct']}% 매수 (없으면 할 일 없음)"
    else:
        head = "관망: 이번 주는 매도도 재매수도 하지 않음"
    change = "" if d["state_kr"] == d["prev_state_kr"] else f" (지난주 {d['prev_state_kr']} -> {d['state_kr']})"
    body = f"{head}{change}\n점수 {d['score']:+.2f} = (금리 {d['rates']:+.2f} + 달러 {d['dxy']:+.2f}) / 2, 기준일 {d['asof']}"
    return f"이번 주 신호: {d['state_kr']}", body


def push(title, body):
    key = os.environ.get("VAPID_PRIVATE_KEY", "").strip()
    subs_raw = os.environ.get("PUSH_SUBSCRIPTIONS", "").strip()
    if not key or not subs_raw:
        print("push skipped: VAPID_PRIVATE_KEY or PUSH_SUBSCRIPTIONS not set")
        return
    from pywebpush import WebPushException, webpush
    subs = json.loads(subs_raw)
    subs = subs if isinstance(subs, list) else [subs]
    claim = os.environ.get("VAPID_SUB", "https://github.com")
    for s in subs:
        try:
            webpush(subscription_info=s, data=json.dumps({"title": title, "body": body, "url": "./"}, ensure_ascii=False),
                    vapid_private_key=key, vapid_claims={"sub": claim}, ttl=86400)
            print("pushed to", s.get("endpoint", "")[:40], "...")
        except WebPushException as e:
            print("push failed:", repr(e)[:200])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()
    d = build(compute())
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    with open(DATA, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)
    title, body = message(d)
    print(title)
    print(body)
    if not args.no_push:
        push(title, body)


if __name__ == "__main__":
    main()
