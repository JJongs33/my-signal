"""Job for the '쫑우 투자신호' app.

Runs twice a day, Tuesday-Saturday, on GitHub Actions: 10:00 Korea time (the US close of the previous US day is in by then) and 21:00 Korea time
(a reminder before the US market opens at 22:30).  There is no new price data between the two runs, so both compute the same score.

1. Computes the rates + dxy score exactly as in the backtest:
     rates = clip( -(10y Treasury yield [1-day lag] - its 126-trading-day mean) / 0.5 , -1, 1 )
     dxy   = clip( -(US dollar index / its 100-trading-day mean - 1) / 0.03 , -1, 1 )
     score = (rates + dxy) / 2      on the NASDAQ-100 trading calendar, read EVERY trading day
   The app shows score x 100 (sell at -30 or lower, buy at 0 or higher).
   Sell strength (added 2026-10-17, twoy_ff_deep.py): spread = 2-year Treasury yield - effective fed funds rate (both 1-day lag).
     If the spread is higher than 252 trading days ago (the market prices more hikes / fewer cuts than a year ago) the sell zone is
     'strong' and sells 5% a week; otherwise it is 'weak' and sells 2.5% a week.
   Buy speed (2026-10-17, sizing_test.py / buy_lower_test.py): in a buy zone, buy 20% of the cash for the first 4 weeks
     (20 trading days), then 40% while the buy zone continues.  A short bounce spends little of the cash; a lasting recovery uses it faster.
2. Writes docs/data.json (signal + 10 years of weekly history) and docs/prices.json (daily QLD / USD closes and USD/KRW).
3. Sends a web-push notification on every run EXCEPT when the state is HOLD and was HOLD the day before:
     "<매도|매수|관망> 구간입니다" / "점수는 -88" (plus "· 관망에서 바뀜" on the first day of a new state).
   The decision uses only the score history (no stored state), so a missed run never loses a state change.

    python signal_job.py                  # compute, write data, push (state changes and sell/buy days)
    python signal_job.py --slot 아침      # same, the slot name is added to the message
    python signal_job.py --no-push        # compute and write only
    python signal_job.py --test-push      # also send the current state once (to test the phone)
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
SELL_PCT, BUY_PCT = 5, 20
BUY_PCT_LATE, BUY_RAMP_DAYS = 40, 20   # after 20 trading days in a buy zone, buy 40% of the cash instead of 20%
SELL_PCT_WEAK = 2.5  # sell zone while the 2y - fed funds spread is not higher than a year ago
FILTER_DAYS = 252    # 'a year ago' in trading days
SCALE = 100        # the app shows score x 100
HISTORY_WEEKS = 530  # about 10 years
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
    y2 = fred("DGS2").reindex(full, method="ffill").shift(1)
    ff = fred("DFF").reindex(full, method="ffill").shift(1)
    sc["y2"], sc["ff"] = y2, ff
    sc["sp"] = y2 - ff
    sc["sp_1y"] = sc["sp"].shift(FILTER_DAYS)
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
    chg = row["sp"] - row["sp_1y"]
    strong = bool(pd.isna(chg) or chg > 0)
    filt = {"strong": strong, "y2": r(row["y2"]), "ff": r(row["ff"]), "spread": r(row["sp"]), "spread_1y": r(row["sp_1y"]),
            "chg": None if pd.isna(chg) else r(chg), "pct_strong": SELL_PCT, "pct_weak": SELL_PCT_WEAK}
    return {
        "updated": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        "asof": str(row.name.date()), "state": row["state"], "state_kr": NAMES[row["state"]], "run": run, "scale": SCALE,
        "score": r(row["score"]), "rates": r(row["rates"]), "dxy": r(row["dxy"]),
        "y10": r(row["y10"]), "y10_mean": r(row["y10_mean"]), "dxy_lvl": r(row["dxy_lvl"]), "dxy_mean": r(row["dxy_mean"]),
        "prev_state_kr": NAMES[prev["state"]], "prev_score": r(prev["score"]),
        "threshold": THRESHOLD, "buy_level": BUY_LEVEL, "sell_pct": SELL_PCT if strong else SELL_PCT_WEAK, "buy_pct": BUY_PCT,
        "sell_filter": filt,
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
        head = f"매도 {d['run']}주째: 다음 거래일에 보유 주식의 {d['sell_pct']}% 매도 -> 현금 보관"
    elif d["state"] == "BUY":
        head = f"매수: 매도로 쌓인 현금이 있으면 그 {d['buy_pct']}% 매수 (없으면 할 일 없음)"
    else:
        head = "관망: 이번 주는 매도도 매수도 하지 않음"
    change = "" if d["state_kr"] == d["prev_state_kr"] else f" (지난주 {d['prev_state_kr']} -> {d['state_kr']})"
    body = (f"{head}{change}\n점수 {f(d['score'])} (매도 {round(d['threshold'] * sc_)} 이하 / 매수 {round(d['buy_level'] * sc_)} 이상), "
            f"기준일 {d['asof']}")
    return f"이번 주 신호: {d['state_kr']}", body


def next_action_in(run_days):
    """0 = act tonight (1st day of the stretch and every 5th trading day after it), else trading days until the next action day."""
    return (CYCLE_DAYS - (run_days - 1) % CYCLE_DAYS) % CYCLE_DAYS


def daily_info(sc, prev=None):
    """-> (daily state dict, (kind, previous state)).  kind: 'enter' (first day of a new state) | 'stay' (inside sell/buy) | None (HOLD continuing)."""
    s = sc["score"].dropna()
    states = [state_of(v) for v in s.to_numpy()]
    cur, asof = states[-1], s.index[-1]
    run_days = 0
    for st in states[::-1]:
        if st != cur:
            break
        run_days += 1
    prev_state = states[-1 - run_days] if run_days < len(states) else None
    kind = "enter" if run_days == 1 else ("stay" if cur in ("SELL", "BUY") else None)
    return {"state": cur, "asof": str(asof.date()), "run_days": run_days}, (kind, prev_state)


def alert_message(d, kind, prev_state):
    """-> (title, body), e.g. ('매도 구간입니다', '점수는 -88 · 매도 강도 강(주 5%) · 관망에서 바뀜')."""
    score = f"{round(d['score'] * d['scale']):+d}"
    word = NAMES[d["state"]]
    body = f"점수는 {score}"
    if d["state"] == "SELL":
        body += f" · 매도 강도 {'강' if d['sell_filter']['strong'] else '약'}(주 {d['sell_pct']:g}%)"
    elif d["state"] == "BUY":
        body += f" · 현금의 {d['buy_pct']:g}% 매수" + (" (4주가 넘어 빨라짐)" if d["buy_pct"] > BUY_PCT else "")
    if kind == "enter" and prev_state and prev_state != d["state"]:
        body += f" · {NAMES[prev_state]}에서 바뀜"
    return f"{word} 구간입니다", body


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
    ap.add_argument("--slot", default="", help="name of this run (e.g. 아침 / 저녁), added to the message")
    args = ap.parse_args()
    prev = None
    try:
        with open(DATA, encoding="utf-8") as f:
            prev = json.load(f)
    except Exception:
        pass
    wk, sc = compute()
    d = build(wk)
    daily, (ev, prev_state) = daily_info(sc, prev)
    d["daily"] = daily
    d["run"] = (daily["run_days"] - 1) // CYCLE_DAYS + 1   # action rounds: day 1, 6, 11, ... of the stretch
    d["run_days"] = daily["run_days"]
    d["buy_pct_early"], d["buy_pct_late"], d["buy_ramp_days"] = BUY_PCT, BUY_PCT_LATE, BUY_RAMP_DAYS
    if d["state"] == "BUY" and daily["run_days"] > BUY_RAMP_DAYS:
        d["buy_pct"] = BUY_PCT_LATE
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    with open(DATA, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, separators=(",", ":"))
    with open(PRICES, "w", encoding="utf-8") as f:
        json.dump(build_prices(sc), f, separators=(",", ":"))
    print(f"asof {d['asof']} state {d['state']} score {round(d['score'] * SCALE)} run_days {daily['run_days']} event {ev} (previous state {prev_state})")
    if args.no_push:
        return
    if ev:
        title, body = alert_message(d, ev, prev_state)
        if args.slot:
            body += f" · {args.slot} 알림"
        print(title); print(body)
        push(title, body)
    elif args.test_push:
        t, b = alert_message(d, None, None)
        push("테스트 알림", f"{t}. {b}")
    else:
        print("no state change / repeat day today: no notification")


if __name__ == "__main__":
    main()
