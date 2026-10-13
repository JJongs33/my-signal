"""Job for the '쫑우 투자신호' app.

Runs every evening at 20:00 Korea time, Tuesday-Saturday (= right after the US market close of the previous US day, GitHub Actions).

1. Computes the rates + dxy score exactly as in the backtest:
     rates = clip( -(10y Treasury yield [1-day lag] - its 126-trading-day mean) / 0.5 , -1, 1 )
     dxy   = clip( -(US dollar index / its 100-trading-day mean - 1) / 0.03 , -1, 1 )
     score = (rates + dxy) / 2      on the NASDAQ-100 trading calendar, read EVERY trading day
   The app shows score x 100 (sell at -30 or lower, buy at 0 or higher).
2. Writes docs/data.json (signal + 10 years of weekly history) and docs/prices.json (daily QLD / USD closes and USD/KRW).
3. Sends a web-push notification ONLY when the score moves INTO the sell zone or INTO the buy zone (secrets VAPID_PRIVATE_KEY and
   PUSH_SUBSCRIPTIONS).  Entering the same zone again within 7 days of the last alert is recorded but not announced (noise at the line).
   The state of the previous run is kept in data.json ("daily"), so a missed run never loses a crossing.
   Action rule that was tested: first action on the NEXT trading day after the crossing, then again every 5th trading day while the
   score stays in the zone (sell 5% of the holding / buy 25% of the cash pile each time).

    python signal_job.py             # compute, write data, push on a crossing
    python signal_job.py --no-push   # compute and write only
    python signal_job.py --test-push # also send the current state once (to test the phone)
"""
import argparse
import io
import json
import os
from datetime import datetime, timezone

import pandas as pd
import requests
import yfinance as yf

THRESHOLD = -0.30  # score at or below: sell 5% of the holding this week
BUY_LEVEL = 0.0    # score at or above: buy with 25% of the cash from earlier sales
SELL_PCT, BUY_PCT = 5, 25
SCALE = 100        # the app shows score x 100
HISTORY_WEEKS = 530  # about 10 years
COOLDOWN_DAYS = 7    # no second alert for the same zone within this many calendar days
CYCLE_DAYS = 5       # actions repeat every 5th trading day while the score stays in the zone
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "docs", "data.json")
PRICES = os.path.join(HERE, "docs", "prices.json")
NAMES = {"SELL": "매도", "BUY": "매수", "HOLD": "관망"}


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
    usd = close("USD").reindex(full, method="ffill")
    krw = close("KRW=X").reindex(full, method="ffill")
    sc = pd.DataFrame(index=full)
    sc["rates"] = (-(y - y.rolling(126).mean()) / 0.5).clip(-1, 1)
    sc["dxy"] = (-(dxy / dxy.rolling(100).mean() - 1) / 0.03).clip(-1, 1)
    sc["score"] = (sc["rates"] + sc["dxy"]) / 2
    sc["y10"], sc["y10_mean"] = y, y.rolling(126).mean()
    sc["dxy_lvl"], sc["dxy_mean"] = dxy, dxy.rolling(100).mean()
    sc["qld"], sc["usd"], sc["krw"] = qld, usd, krw
    iso = sc.index.isocalendar()
    wk = sc.groupby([iso.year.values, iso.week.values]).tail(1).dropna(subset=["score"])
    return wk, sc


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
    hist = wk.iloc[-HISTORY_WEEKS:]
    changes = []
    st = wk["state"].to_numpy()
    for i in range(1, len(wk)):
        if st[i] != st[i - 1] and wk.index[i] >= wk.index[-1] - pd.Timedelta(days=5 * 365):
            changes.append({"d": str(wk.index[i].date()), "from": NAMES[st[i - 1]], "to": NAMES[st[i]], "s": round(float(wk["score"].iloc[i]), 3)})
    r = lambda x: round(float(x), 3)
    return {
        "updated": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        "asof": str(row.name.date()), "state": row["state"], "state_kr": NAMES[row["state"]], "run": run, "scale": SCALE,
        "score": r(row["score"]), "rates": r(row["rates"]), "dxy": r(row["dxy"]),
        "y10": r(row["y10"]), "y10_mean": r(row["y10_mean"]), "dxy_lvl": r(row["dxy_lvl"]), "dxy_mean": r(row["dxy_mean"]),
        "prev_state_kr": NAMES[prev["state"]], "prev_score": r(prev["score"]),
        "threshold": THRESHOLD, "buy_level": BUY_LEVEL, "sell_pct": SELL_PCT, "buy_pct": BUY_PCT,
        "history": [{"d": str(d.date()), "s": round(float(s), 3), "q": round(float(q), 2), "u": round(float(u), 2)}
                    for d, s, q, u in zip(hist.index, hist["score"], hist["qld"], hist["usd"])],
        "qld_now": round(float(row["qld"]), 2), "usd_now": round(float(row["usd"]), 2),
        "changes": changes[::-1],
    }


def build_prices(sc):
    """Daily closes (QLD, USD fund) and USD/KRW for about 10.5 years, so the app can price a purchase on any date."""
    d = sc[["qld", "usd", "krw"]].dropna().loc[pd.Timestamp.now().normalize() - pd.Timedelta(days=int(10.6 * 365)):]
    return {"updated": datetime.now(timezone.utc).isoformat(timespec="minutes"),
            "rows": [[str(i.date()), round(float(a), 2), round(float(b), 2), round(float(c), 1)] for i, a, b, c in zip(d.index, d["qld"], d["usd"], d["krw"])]}


def message(d):
    sc_ = d["scale"]
    f = lambda x: f"{round(x * sc_):+d}"
    if d["state"] == "SELL":
        head = f"매도 {d['run']}주째: 다음 거래일에 보유분의 {d['sell_pct']}% 매도 -> 현금 보관"
    elif d["state"] == "BUY":
        head = f"매수: 매도로 쌓인 현금이 있으면 그 {d['buy_pct']}% 매수 (없으면 할 일 없음)"
    else:
        head = "관망: 이번 주는 매도도 매수도 하지 않음"
    change = "" if d["state_kr"] == d["prev_state_kr"] else f" (지난주 {d['prev_state_kr']} -> {d['state_kr']})"
    body = (f"{head}{change}\n점수 {f(d['score'])} (매도 {round(d['threshold'] * sc_)} 이하 / 매수 {round(d['buy_level'] * sc_)} 이상), "
            f"기준일 {d['asof']}")
    return f"이번 주 신호: {d['state_kr']}", body


def daily_info(sc, prev):
    """Crossing detection on the daily score. prev = the previous data.json (dict) or None."""
    s = sc["score"].dropna()
    states = [state_of(v) for v in s.to_numpy()]
    cur, asof = states[-1], s.index[-1]
    run_days = 0
    for st in states[::-1]:
        if st != cur:
            break
        run_days += 1
    old = (prev or {}).get("daily") or {}
    last_alert = dict(old.get("last_alert") or {"SELL": None, "BUY": None})
    alert = None
    if old.get("state") and old["state"] != cur and str(asof.date()) > old.get("asof", ""):
        if cur in ("SELL", "BUY"):
            la = last_alert.get(cur)
            if la is None or (asof - pd.Timestamp(la)).days > COOLDOWN_DAYS:
                alert = cur
                last_alert[cur] = str(asof.date())
            else:
                print(f"crossing into {cur} on {asof.date()} not announced (last {cur} alert {la}, within {COOLDOWN_DAYS} days)")
    return {"state": cur, "asof": str(asof.date()), "run_days": run_days, "last_alert": last_alert}, alert


def alert_message(d, kind):
    sc_ = d["scale"]
    score = f"{round(d['score'] * sc_):+d}"
    if kind == "SELL":
        return ("매도 신호가 켜졌어요",
                f"점수 {score}: 매도 구간(-30 이하)으로 넘어갔습니다. 오늘 미국 장에서 보유분의 {d['sell_pct']}%를 매도해 현금으로 보관하세요. "
                f"구간이 이어지면 5거래일(약 1주)마다 반복합니다. (기준일 {d['asof']})")
    return ("매수 신호가 켜졌어요",
            f"점수 {score}: 매수 구간(0 이상)으로 넘어갔습니다. 매도로 모아 둔 현금이 있으면 오늘 미국 장에서 그 {d['buy_pct']}%를 매수하세요 "
            f"(현금이 없으면 할 일 없음). 구간이 이어지면 5거래일(약 1주)마다 반복합니다. (기준일 {d['asof']})")


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
    ap.add_argument("--test-push", action="store_true", help="also send the current state once, to test the phone")
    args = ap.parse_args()
    prev = None
    try:
        with open(DATA, encoding="utf-8") as f:
            prev = json.load(f)
    except Exception:
        pass
    wk, sc = compute()
    d = build(wk)
    daily, alert = daily_info(sc, prev)
    d["daily"] = daily
    d["run"] = (daily["run_days"] - 1) // CYCLE_DAYS + 1   # action rounds: day 1, 6, 11, ... of the stretch
    d["run_days"] = daily["run_days"]
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    with open(DATA, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, separators=(",", ":"))
    with open(PRICES, "w", encoding="utf-8") as f:
        json.dump(build_prices(sc), f, separators=(",", ":"))
    print(f"asof {d['asof']} state {d['state']} score {round(d['score'] * SCALE)} run_days {daily['run_days']} alert {alert}")
    if args.no_push:
        return
    if alert:
        title, body = alert_message(d, alert)
        print(title); print(body)
        push(title, body)
    elif args.test_push:
        push("테스트 알림", f"지금 상태: {d['state_kr']} (점수 {round(d['score'] * SCALE):+d}). 점수가 매도·매수 구간으로 넘어가는 날 저녁 8시(한국 시간)에 알림이 옵니다.")
    else:
        print("no zone crossing today: no notification")


if __name__ == "__main__":
    main()
