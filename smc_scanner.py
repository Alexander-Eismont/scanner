"""SMC/ICT scanner: Binance TOP USDT Spot universe + TradingView OHLCV + Telegram.

Python >= 3.11

The Binance public REST API is used ONLY to discover/rank active USDT spot pairs
by 24h quote volume. Historical OHLCV used by the SMC engine comes from TradingView
through tvkit. No orders are placed.
"""

from __future__ import annotations

import argparse
import asyncio
import html
import logging
import os
import re
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any

import aiohttp
import numpy as np
import pandas as pd
import pyvsmc as smc
from dotenv import load_dotenv
from tvkit.api.chart.ohlcv import OHLCV


# ============================================================
# 1. CONFIG
# ============================================================

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# Universe: TOP N активных Binance Spot пар с котировкой USDT.
TOP_N_SYMBOLS = int(os.getenv("TOP_N_SYMBOLS", "70"))
UNIVERSE_REFRESH_SECONDS = int(os.getenv("UNIVERSE_REFRESH_SECONDS", str(6 * 60 * 60)))
MIN_24H_QUOTE_VOLUME = float(os.getenv("MIN_24H_QUOTE_VOLUME", "0"))

# Для входов используются оба ТФ. На 1H направление задаёт 4H.
TIMEFRAMES: dict[str, str] = {"1H": "60", "4H": "240"}
INTERVAL_SECONDS = {"1H": 60 * 60, "4H": 4 * 60 * 60}
ENTRY_TIMEFRAME = "1H"

BARS_COUNT = int(os.getenv("BARS_COUNT", "500"))
CLOSED_CANDLES_ONLY = True

# Ограничения запросов: 80 серий на 40 инструментах, но не более N одновременно.
MAX_TV_CONCURRENT_REQUESTS = int(os.getenv("MAX_TV_CONCURRENT_REQUESTS", "3"))
SCAN_INTERVAL_SECONDS = int(os.getenv("SCAN_INTERVAL_SECONDS", "600"))  # 10 минут

# SMC параметры.
SWING_WINDOW = 3
SWEEP_LOOKBACK = 20
SWEEP_MAX_AGE = 10
STRUCTURE_MAX_AGE = 10
OB_LOOKBACK = 15
MAX_ZONE_AGE = 40
UNICORN_CONFLUENCE_BARS = 4
RANGE_LOOKBACK = 40
MIN_RR = 2.0
SL_BUFFER = 0.001  # 0.1% за экстремумом/зоной

# Необязательный расчёт размера позиции. Ноль = отключено.
# Вводится вручную; скрипт НЕ получает баланс биржевого аккаунта.
ACCOUNT_EQUITY_USDT = float(os.getenv("ACCOUNT_EQUITY_USDT", "0"))
RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "1.0"))

# Binance market-data-only endpoint; fallback на основной API.
BINANCE_API_BASES = (
    "https://data-api.binance.vision",
    "https://api.binance.com",
)
BINANCE_EXCHANGE_INFO_PATH = "/api/v3/exchangeInfo"
BINANCE_TICKER_24H_PATH = "/api/v3/ticker/24hr"

STABLE_BASE_ASSETS = {
    "USDT", "USDC", "BUSD", "FDUSD", "TUSD", "USDP", "DAI",
    "USDE", "USDS", "UST", "USTC", "PYUSD", "USD1", "EURC",
    "RLUSD", "U",
}

# Активы, которые не должны попадать в список обычных криптовалют:
# RWUSD — продукт Binance Earn; EUR — фиат; XAUT/PAXG — золото-токены.
NON_CRYPTO_BASE_ASSETS = {"RWUSD", "EUR", "XAUT", "PAXG"}

# Binance bStocks — токенизированные ценные бумаги; их тикеры оканчиваются на B.
# BNB оставляем, так как это обычная криптовалюта.
LEVERAGED_TOKEN_SUFFIX = re.compile(r"(?:UP|DOWN|BULL|BEAR)$", re.IGNORECASE)

DATABASE_PATH = os.getenv("SIGNAL_DB_PATH", "smc_signals.sqlite3")
TELEGRAM_API_URL = (
    f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    if TELEGRAM_TOKEN else ""
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("smc_scanner.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("SMC_SCANNER")

# Кэш состава рынка в памяти текущего процесса.
_UNIVERSE_CACHE: list[str] = []
_UNIVERSE_CACHE_TS = 0.0


# ============================================================
# 2. SQLITE: persistent duplicate protection
# ============================================================

def init_database() -> None:
    """Создаёт таблицу для защиты от повторных уведомлений после перезапуска."""
    with sqlite3.connect(DATABASE_PATH, timeout=10) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sent_signals (
                signal_key TEXT PRIMARY KEY,
                sent_at_utc TEXT NOT NULL
            )
            """
        )
        conn.commit()


def signal_was_sent(signal_key: str) -> bool:
    with sqlite3.connect(DATABASE_PATH, timeout=10) as conn:
        row = conn.execute(
            "SELECT 1 FROM sent_signals WHERE signal_key = ? LIMIT 1",
            (signal_key,),
        ).fetchone()
    return row is not None


def remember_signal(signal_key: str) -> None:
    with sqlite3.connect(DATABASE_PATH, timeout=10) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO sent_signals(signal_key, sent_at_utc) VALUES (?, ?)",
            (signal_key, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()


# ============================================================
# 3. MARKET UNIVERSE: TOP N Binance USDT Spot
# ============================================================

async def _get_json_from_binance(
    session: aiohttp.ClientSession,
    path: str,
) -> Any:
    """Запрашивает публичные Binance market-data endpoints с fallback-хостом."""
    last_error: Exception | None = None

    for base_url in BINANCE_API_BASES:
        try:
            async with session.get(f"{base_url}{path}") as response:
                body = await response.text()
                if response.status != 200:
                    raise RuntimeError(
                        f"Binance HTTP {response.status}: {body[:300]}"
                    )
                import json
                return json.loads(body)
        except Exception as exc:  # noqa: BLE001 - fallback between official hosts
            last_error = exc
            logger.warning("Binance endpoint %s failed: %s", base_url, exc)

    raise RuntimeError(f"Не удалось получить данные Binance: {last_error}")


async def get_top_usdt_symbols(
    session: aiohttp.ClientSession,
    force_refresh: bool = False,
) -> list[str]:
    """Возвращает TOP N Binance Spot USDT-пар по 24h quoteVolume.

    Binance нужен только для отбора тикеров. OHLCV далее загружается из TradingView.
    """
    global _UNIVERSE_CACHE, _UNIVERSE_CACHE_TS

    now = time.time()
    if (
        not force_refresh
        and _UNIVERSE_CACHE
        and now - _UNIVERSE_CACHE_TS < UNIVERSE_REFRESH_SECONDS
    ):
        return list(_UNIVERSE_CACHE)

    exchange_info, tickers = await asyncio.gather(
        _get_json_from_binance(session, BINANCE_EXCHANGE_INFO_PATH),
        _get_json_from_binance(session, BINANCE_TICKER_24H_PATH),
    )

    if not isinstance(exchange_info, dict) or not isinstance(tickers, list):
        raise RuntimeError("Неожиданный формат ответа Binance API")

    eligible: set[str] = set()
    for pair in exchange_info.get("symbols", []):
        if pair.get("status") != "TRADING":
            continue
        if pair.get("quoteAsset") != "USDT":
            continue
        if not pair.get("isSpotTradingAllowed", True):
            continue

        base = str(pair.get("baseAsset", "")).upper()
        symbol = str(pair.get("symbol", "")).upper()

        if not base or base in STABLE_BASE_ASSETS:
            continue
        if base in NON_CRYPTO_BASE_ASSETS:
            continue
        if base.endswith("B") and base != "BNB":
            # Исключаем Binance bStocks (NVDAB, GOOGLB, MSTRB, SOXLB и т.п.).
            continue
        if LEVERAGED_TOKEN_SUFFIX.search(base):
            continue
        if not symbol.endswith("USDT"):
            continue

        eligible.add(symbol)

    volume_by_symbol: list[tuple[str, float]] = []
    for ticker in tickers:
        symbol = str(ticker.get("symbol", "")).upper()
        if symbol not in eligible:
            continue
        try:
            quote_volume = float(ticker.get("quoteVolume", 0.0))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(quote_volume) or quote_volume < MIN_24H_QUOTE_VOLUME:
            continue
        volume_by_symbol.append((symbol, quote_volume))

    volume_by_symbol.sort(key=lambda item: item[1], reverse=True)
    selected = [f"BINANCE:{symbol}" for symbol, _ in volume_by_symbol[:TOP_N_SYMBOLS]]

    if not selected:
        raise RuntimeError("Binance не вернул подходящих активных USDT Spot пар")

    _UNIVERSE_CACHE = selected
    _UNIVERSE_CACHE_TS = now

    logger.info(
        "Universe обновлён: %d/%d инструментов (Binance Spot USDT, TOP по 24h quote volume)",
        len(selected),
        TOP_N_SYMBOLS,
    )
    logger.info("Инструменты: %s", ", ".join(s.split(":", 1)[-1] for s in selected))
    return list(selected)


# ============================================================
# 4. TRADINGVIEW OHLCV
# ============================================================

async def fetch_data(
    symbol: str,
    timeframe: str,
    semaphore: asyncio.Semaphore,
) -> pd.DataFrame | None:
    """Асинхронно получает исторические OHLCV-свечи через tvkit."""
    if timeframe not in TIMEFRAMES:
        raise ValueError(f"Неподдерживаемый timeframe: {timeframe}")

    try:
        async with semaphore:
            # tvkit автоматически использует TVKIT_AUTH_TOKEN или TVKIT_BROWSER
            # из окружения; без них работает анонимная сессия.
            async with OHLCV() as client:
                bars = await client.get_historical_ohlcv(
                    exchange_symbol=symbol,
                    interval=TIMEFRAMES[timeframe],
                    bars_count=BARS_COUNT,
                )

        if not bars:
            logger.warning("%s %s: TradingView вернул пустые данные", symbol, timeframe)
            return None

        df = pd.DataFrame(
            [
                {
                    "timestamp": float(bar.timestamp),
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": float(bar.volume),
                }
                for bar in bars
            ]
        )

        df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)

        price_cols = ["open", "high", "low", "close"]
        df[price_cols] = df[price_cols].apply(pd.to_numeric, errors="coerce")
        df = df.dropna(subset=price_cols)
        df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)]
        df = df[df["high"] >= df["low"]]

        # Не удаляем последнюю свечу вслепую: убираем её только если её
        # интервал ещё не завершился по UTC.
        if CLOSED_CANDLES_ONLY:
            now_ts = datetime.now(timezone.utc).timestamp()
            open_ts = df["timestamp"].astype("int64") / 1_000_000_000
            df = df[open_ts + INTERVAL_SECONDS[timeframe] <= now_ts + 1]

        df = df.reset_index(drop=True)
        if len(df) < 100:
            logger.warning("%s %s: мало закрытых свечей (%d)", symbol, timeframe, len(df))
            return None

        logger.debug("%s %s: загружено %d закрытых свечей", symbol, timeframe, len(df))
        return df

    except Exception as exc:  # noqa: BLE001
        logger.warning("Ошибка TradingView %s %s: %s", symbol, timeframe, exc)
        return None


# ============================================================
# 5. PYVSMC ANALYSIS CONTEXT
# ============================================================

def analyze_smc(df: pd.DataFrame) -> dict[str, Any]:
    """Считает все SMC-компоненты pyvsmc на одной и той же истории свечей."""
    if df is None or len(df) < 100:
        raise ValueError("Недостаточно OHLCV данных для SMC")

    o = df["open"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)

    swings = smc.detect_swings(
        high=h,
        low=l,
        window_size=SWING_WINDOW,
        tie="first",
    )

    structure = smc.detect_structure(
        high=h,
        low=l,
        close=c,
        window_size=SWING_WINDOW,
        break_mode="close",
        tie="first",
        swing_high=swings.swing_high,
        swing_low=swings.swing_low,
    )

    liquidity = smc.detect_liquidity(
        high=h,
        low=l,
        close=c,
        equal_threshold=0.001,
        sweep_lookback=SWEEP_LOOKBACK,
        window_size=SWING_WINDOW,
        tie="first",
        swing_high=swings.swing_high,
        swing_low=swings.swing_low,
    )

    order_blocks = smc.detect_order_blocks(
        open_=o,
        high=h,
        low=l,
        close=c,
        lookback=OB_LOOKBACK,
        use_fvg=True,
        use_bos=True,
        window_size=SWING_WINDOW,
        zone_mode="full",
        compute_mitigation=True,
        tie="first",
        break_mode="close",
    )

    fvg = smc.detect_fvg(
        high=h,
        low=l,
        compute_mitigation=True,
    )

    zones = smc.detect_zones(
        high=h,
        low=l,
        close=c,
        window_size=SWING_WINDOW,
        eq_threshold=0.02,
        tie="first",
        swing_high=swings.swing_high,
        swing_low=swings.swing_low,
    )

    return {
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "swings": swings,
        "structure": structure,
        "liquidity": liquidity,
        "ob": order_blocks,
        "fvg": fvg,
        "zones": zones,
        "trend": int(structure.trend[-1]),
        "price": float(c[-1]),
        "index": len(df) - 1,
    }


# ============================================================
# 6. COMMON SMC HELPERS
# ============================================================

def last_true(values: np.ndarray, start: int = 0, end: int | None = None) -> int | None:
    arr = np.asarray(values, dtype=bool)
    if end is None:
        end = len(arr)
    start = max(0, start)
    end = min(len(arr), end)
    if start >= end:
        return None
    indexes = np.flatnonzero(arr[start:end])
    return int(indexes[-1] + start) if indexes.size else None


def candle_intersects_zone(ctx: dict[str, Any], index: int, lower: float, upper: float) -> bool:
    return bool(ctx["low"][index] <= upper and ctx["high"][index] >= lower)


def get_active_fvg(
    ctx: dict[str, Any],
    direction: str,
    after_index: int = 0,
) -> dict[str, Any] | None:
    """Находит свежий FVG, куда вернулась текущая свеча.

    Частичный ретест допускается. Полностью заполненный FVG не используется.
    Если FVG был впервые затронут раньше текущей свечи, он уже не считается fresh.
    """
    fvg = ctx["fvg"]
    n = ctx["index"]

    if direction == "LONG":
        mask = np.asarray(fvg.bullish, dtype=bool)
        lower_arr = fvg.bullish_lower
        upper_arr = fvg.bullish_upper
    else:
        mask = np.asarray(fvg.bearish, dtype=bool)
        lower_arr = fvg.bearish_lower
        upper_arr = fvg.bearish_upper

    mitigated = np.asarray(fvg.mitigated, dtype=bool)
    mitigated_idx = np.asarray(fvg.mitigated_index, dtype=int)
    fully_filled = np.asarray(fvg.mitigated_full, dtype=bool)

    candidate_mask = mask & (np.arange(len(mask)) >= after_index)
    for idx in np.flatnonzero(candidate_mask)[::-1]:
        if n - int(idx) > MAX_ZONE_AGE:
            continue
        if fully_filled[idx]:
            continue
        if mitigated[idx] and mitigated_idx[idx] != n:
            continue

        lower = float(lower_arr[idx])
        upper = float(upper_arr[idx])
        if not np.isfinite(lower) or not np.isfinite(upper):
            continue
        if lower > upper:
            lower, upper = upper, lower

        if candle_intersects_zone(ctx, n, lower, upper):
            return {
                "lower": lower,
                "upper": upper,
                "index": int(idx),
                "event_index": int(idx),
                "type": "FVG",
            }

    return None


def get_active_ob(
    ctx: dict[str, Any],
    direction: str,
    after_index: int = 0,
) -> dict[str, Any] | None:
    """Находит свежий OB по validated_index, а не по свече-якорю OB.

    Это важно: Order Block часто расположен ДО BOS/CHoCH, а подтверждается
    импульсом BOS/CHoCH. Также допускается первый ретест OB на текущей свече.
    """
    ob = ctx["ob"]
    n = ctx["index"]

    mask = np.asarray(ob.bullish_ob if direction == "LONG" else ob.bearish_ob, dtype=bool)
    validated = np.asarray(ob.validated_index, dtype=int)
    mitigated = np.asarray(ob.mitigated, dtype=bool)
    mitigated_idx = np.asarray(ob.mitigated_index, dtype=int)
    breakers = np.asarray(ob.is_breaker, dtype=bool)

    candidates = np.flatnonzero(
        mask
        & (validated >= after_index)
        & (validated >= 0)
        & (validated <= n)
    )
    candidates = sorted(candidates, key=lambda i: validated[i], reverse=True)

    for idx in candidates:
        if n - int(validated[idx]) > MAX_ZONE_AGE:
            continue
        if breakers[idx]:
            continue
        if mitigated[idx] and mitigated_idx[idx] != n:
            continue

        lower = float(ob.ob_low[idx])
        upper = float(ob.ob_high[idx])
        if not np.isfinite(lower) or not np.isfinite(upper):
            continue
        if lower > upper:
            lower, upper = upper, lower

        if candle_intersects_zone(ctx, n, lower, upper):
            return {
                "lower": lower,
                "upper": upper,
                "index": int(idx),             # свеча-якорь OB
                "event_index": int(validated[idx]),  # импульс, подтвердивший OB
                "type": "OB",
            }

    return None


def get_entry_zone(
    ctx: dict[str, Any],
    direction: str,
    after_index: int = 0,
) -> dict[str, Any] | None:
    """Выбирает самый свежий из актуальных FVG и OB, затронутых текущей свечой."""
    candidates = [
        z for z in (
            get_active_fvg(ctx, direction, after_index),
            get_active_ob(ctx, direction, after_index),
        ) if z is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda zone: zone["event_index"])


def nearest_liquidity_target(ctx: dict[str, Any], direction: str, entry: float) -> float | None:
    """Ближайший подтверждённый swing high для LONG или swing low для SHORT."""
    swings = ctx["swings"]
    n = ctx["index"]
    confirmed_before = max(0, n - SWING_WINDOW)

    if direction == "LONG":
        indexes = np.flatnonzero(np.asarray(swings.swing_high, dtype=bool))
        levels = [
            float(ctx["high"][i])
            for i in indexes
            if i < confirmed_before and float(ctx["high"][i]) > entry
        ]
        return min(levels) if levels else None

    indexes = np.flatnonzero(np.asarray(swings.swing_low, dtype=bool))
    levels = [
        float(ctx["low"][i])
        for i in indexes
        if i < confirmed_before and float(ctx["low"][i]) < entry
    ]
    return max(levels) if levels else None


def build_signal(
    ctx: dict[str, Any],
    direction: str,
    zone: dict[str, Any],
    model_name: str,
    reason: str,
    sl_override: float | None = None,
    tp_override: float | None = None,
) -> dict[str, Any] | None:
    """Строит Entry/SL/TP и проверяет минимальный Risk/Reward."""
    if zone is None:
        return None

    lower = float(zone["lower"])
    upper = float(zone["upper"])
    if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
        return None

    entry = (lower + upper) / 2.0
    if entry <= 0:
        return None

    if direction == "LONG":
        sl = float(sl_override) if sl_override is not None else lower * (1 - SL_BUFFER)
        if sl >= entry:
            return None
        risk = entry - sl
        tp = float(tp_override) if tp_override is not None else nearest_liquidity_target(ctx, direction, entry)
        if tp is None:
            tp = entry + risk * MIN_RR
        if tp <= entry:
            return None
    elif direction == "SHORT":
        sl = float(sl_override) if sl_override is not None else upper * (1 + SL_BUFFER)
        if sl <= entry:
            return None
        risk = sl - entry
        tp = float(tp_override) if tp_override is not None else nearest_liquidity_target(ctx, direction, entry)
        if tp is None:
            tp = entry - risk * MIN_RR
        if tp >= entry:
            return None
    else:
        return None

    reward = abs(tp - entry)
    rr = reward / risk if risk > 0 else 0.0
    if rr < MIN_RR:
        return None

    # Это расстояние цены до SL, а не процент убытка от депозита.
    price_risk_percent = risk / entry * 100.0

    # Опциональный sizing: сколько единиц купить/продать, чтобы
    # расстояние Entry->SL соответствовало заданному бюджету риска.
    # Комиссии, funding, проскальзывание и шаг количества биржи не учтены.
    sizing = None
    if ACCOUNT_EQUITY_USDT > 0 and RISK_PER_TRADE_PCT > 0:
        risk_budget_usdt = ACCOUNT_EQUITY_USDT * RISK_PER_TRADE_PCT / 100.0
        quantity = risk_budget_usdt / risk
        notional_usdt = quantity * entry
        sizing = {
            "account_equity_usdt": float(ACCOUNT_EQUITY_USDT),
            "risk_budget_usdt": float(risk_budget_usdt),
            "risk_per_trade_pct": float(RISK_PER_TRADE_PCT),
            "quantity": float(quantity),
            "notional_usdt": float(notional_usdt),
            "equity_multiple": float(notional_usdt / ACCOUNT_EQUITY_USDT),
        }

    return {
        "direction": direction,
        "entry": float(entry),
        "sl": float(sl),
        "tp": float(tp),
        "price_risk_percent": float(price_risk_percent),
        "rr": float(rr),
        "sizing": sizing,
        "model_name": model_name,
        "reason": reason,
        "zone": str(zone["type"]),
        "bar_time": None,
    }


# ============================================================
# 7. MODEL: TREND CONTINUATION
# ============================================================

def model_continuation(
    df: pd.DataFrame,
    higher_df: pd.DataFrame | None,
    ctx: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """4H trend + fresh 1H BOS + retest of the OB/FVG from that impulse."""
    if higher_df is None:
        return None
    ctx = ctx or analyze_smc(df)
    htf = analyze_smc(higher_df)
    trend = htf["trend"]
    if trend == 0:
        return None

    structure = ctx["structure"]
    n = ctx["index"]
    direction = "LONG" if trend == 1 else "SHORT"
    bos_mask = structure.bos_bullish if direction == "LONG" else structure.bos_bearish
    bos_idx = last_true(bos_mask, max(0, n - STRUCTURE_MAX_AGE + 1), n + 1)
    if bos_idx is None or n - bos_idx > STRUCTURE_MAX_AGE:
        return None

    zone = get_entry_zone(ctx, direction, after_index=bos_idx)
    if zone is None:
        return None

    zones = ctx["zones"]
    location_ok = (
        bool(zones.discount[-1] or zones.in_ote[-1])
        if direction == "LONG"
        else bool(zones.premium[-1] or zones.in_ote[-1])
    )
    if not location_ok:
        return None

    trend_text = "Bullish" if trend == 1 else "Bearish"
    bos_text = "Bullish BOS" if direction == "LONG" else "Bearish BOS"
    location_text = "Discount/OTE" if direction == "LONG" else "Premium/OTE"
    return build_signal(
        ctx,
        direction,
        zone,
        "Trend Continuation",
        f"4H {trend_text} trend + 1H {bos_text} + fresh {zone['type']} retest + {location_text}",
    )


# ============================================================
# 8. REVERSAL SETUP: SWEEP + CHOCH
# ============================================================

def find_reversal_setup(ctx: dict[str, Any]) -> dict[str, Any] | None:
    """Находит актуальную последовательность: sweep, затем CHoCH в обратную сторону."""
    liq = ctx["liquidity"]
    structure = ctx["structure"]
    n = ctx["index"]
    start = max(0, n - SWEEP_MAX_AGE)
    setups: list[dict[str, Any]] = []

    # Sweep SSL -> bullish CHoCH -> LONG.
    sweep_lows = np.flatnonzero(np.asarray(liq.sweep_low, dtype=bool)[start:n + 1]) + start
    choch_bull = np.asarray(structure.choch_bullish, dtype=bool)
    for sweep_idx in sweep_lows[::-1]:
        choch_idx = last_true(choch_bull, int(sweep_idx) + 1, n + 1)
        if choch_idx is not None and n - int(sweep_idx) <= SWEEP_MAX_AGE:
            if n - choch_idx <= STRUCTURE_MAX_AGE:
                setups.append({
                    "direction": "LONG",
                    "sweep_index": int(sweep_idx),
                    "choch_index": int(choch_idx),
                    "reason": "Sweep SSL + Bullish CHoCH",
                })
                break

    # Sweep BSL -> bearish CHoCH -> SHORT.
    sweep_highs = np.flatnonzero(np.asarray(liq.sweep_high, dtype=bool)[start:n + 1]) + start
    choch_bear = np.asarray(structure.choch_bearish, dtype=bool)
    for sweep_idx in sweep_highs[::-1]:
        choch_idx = last_true(choch_bear, int(sweep_idx) + 1, n + 1)
        if choch_idx is not None and n - int(sweep_idx) <= SWEEP_MAX_AGE:
            if n - choch_idx <= STRUCTURE_MAX_AGE:
                setups.append({
                    "direction": "SHORT",
                    "sweep_index": int(sweep_idx),
                    "choch_index": int(choch_idx),
                    "reason": "Sweep BSL + Bearish CHoCH",
                })
                break

    return max(setups, key=lambda x: x["choch_index"]) if setups else None


# ============================================================
# 9. MODEL: REVERSAL VIA SWEEP
# ============================================================

def model_reversal(df: pd.DataFrame, ctx: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Sweep key liquidity -> CHoCH -> retest of OB/FVG created/validated by impulse."""
    ctx = ctx or analyze_smc(df)
    setup = find_reversal_setup(ctx)
    if setup is None:
        return None

    zone = get_entry_zone(
        ctx,
        setup["direction"],
        after_index=setup["choch_index"],
    )
    if zone is None:
        return None

    return build_signal(
        ctx,
        setup["direction"],
        zone,
        "Reversal via Sweep",
        f"{setup['reason']} + retest of post-CHoCH {zone['type']}",
    )


# ============================================================
# 10. MODEL: ICT UNICORN (BREAKER + FVG OVERLAP)
# ============================================================

def model_unicorn(df: pd.DataFrame, ctx: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Sweep + CHoCH, then breaker invalidation and overlapping same-direction FVG.

    For LONG: a former bearish OB is broken upwards and becomes bullish breaker.
    For SHORT: a former bullish OB is broken downwards and becomes bearish breaker.
    """
    ctx = ctx or analyze_smc(df)
    setup = find_reversal_setup(ctx)
    if setup is None:
        return None

    direction = setup["direction"]
    n = ctx["index"]
    ob = ctx["ob"]
    fvg = ctx["fvg"]

    breaker_flags = np.asarray(ob.is_breaker, dtype=bool)
    breaker_times = np.asarray(ob.breaker_index, dtype=int)
    ob_types = np.asarray(ob.ob_type, dtype=object)
    expected_original_ob_type = "bearish" if direction == "LONG" else "bullish"

    breaker_candidates = np.flatnonzero(
        breaker_flags
        & (breaker_times >= setup["choch_index"])
        & (breaker_times <= n)
        & (ob_types == expected_original_ob_type)
    )
    if breaker_candidates.size == 0:
        return None

    if direction == "LONG":
        fvg_mask = np.asarray(fvg.bullish, dtype=bool)
        fvg_lower = np.asarray(fvg.bullish_lower, dtype=float)
        fvg_upper = np.asarray(fvg.bullish_upper, dtype=float)
    else:
        fvg_mask = np.asarray(fvg.bearish, dtype=bool)
        fvg_lower = np.asarray(fvg.bearish_lower, dtype=float)
        fvg_upper = np.asarray(fvg.bearish_upper, dtype=float)

    fvg_full = np.asarray(fvg.mitigated_full, dtype=bool)
    fvg_mitigated = np.asarray(fvg.mitigated, dtype=bool)
    fvg_mitigated_idx = np.asarray(fvg.mitigated_index, dtype=int)

    for ob_idx in sorted(
        breaker_candidates,
        key=lambda i: breaker_times[i],
        reverse=True,
    ):
        breaker_idx = int(breaker_times[ob_idx])
        if breaker_idx - setup["choch_index"] > UNICORN_CONFLUENCE_BARS:
            continue
        if n - breaker_idx > MAX_ZONE_AGE:
            continue

        ob_low = float(ob.ob_low[ob_idx])
        ob_high = float(ob.ob_high[ob_idx])
        if not np.isfinite(ob_low) or not np.isfinite(ob_high):
            continue
        if ob_low > ob_high:
            ob_low, ob_high = ob_high, ob_low

        fvg_candidates = np.flatnonzero(
            fvg_mask
            & (np.abs(np.arange(len(fvg_mask)) - breaker_idx) <= UNICORN_CONFLUENCE_BARS)
            & (np.arange(len(fvg_mask)) >= setup["choch_index"])
        )

        for fvg_idx in fvg_candidates[::-1]:
            if fvg_full[fvg_idx]:
                continue
            if fvg_mitigated[fvg_idx] and fvg_mitigated_idx[fvg_idx] != n:
                continue

            fl = float(fvg_lower[fvg_idx])
            fu = float(fvg_upper[fvg_idx])
            if not np.isfinite(fl) or not np.isfinite(fu):
                continue

            overlap_lower = max(ob_low, fl)
            overlap_upper = min(ob_high, fu)
            if overlap_lower >= overlap_upper:
                continue
            if not candle_intersects_zone(ctx, n, overlap_lower, overlap_upper):
                continue

            zone = {
                "lower": overlap_lower,
                "upper": overlap_upper,
                "index": int(fvg_idx),
                "event_index": max(breaker_idx, int(fvg_idx)),
                "type": "Breaker + FVG",
            }
            return build_signal(
                ctx,
                direction,
                zone,
                "ICT Unicorn",
                f"{setup['reason']} + {direction.title()} Breaker Block + same-impulse FVG overlap",
            )

    return None


# ============================================================
# 11. MODEL: RANGE DEVIATION
# ============================================================

def model_range(df: pd.DataFrame, ctx: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Ложный пробой диапазона с закрытием обратно внутрь.

    Текущая свеча не участвует в расчёте границ диапазона (нет look-ahead).
    Цель — противоположная граница диапазона.
    """
    ctx = ctx or analyze_smc(df)
    n = ctx["index"]
    if len(df) < RANGE_LOOKBACK + 5:
        return None

    previous = df.iloc[n - RANGE_LOOKBACK:n]
    range_high = float(previous["high"].max())
    range_low = float(previous["low"].min())

    current_high = float(ctx["high"][n])
    current_low = float(ctx["low"][n])
    current_close = float(ctx["close"][n])

    # LONG: sweep нижней границы и закрытие обратно в диапазоне.
    if current_low < range_low and range_low < current_close < range_high:
        zone = {
            "lower": current_close,
            "upper": current_close,
            "index": n,
            "event_index": n,
            "type": "Range Deviation",
        }
        return build_signal(
            ctx,
            "LONG",
            zone,
            "Range Deviation",
            "SSL/range-low sweep + close back inside range",
            sl_override=current_low * (1 - SL_BUFFER),
            tp_override=range_high,
        )

    # SHORT: sweep верхней границы и закрытие обратно в диапазоне.
    if current_high > range_high and range_low < current_close < range_high:
        zone = {
            "lower": current_close,
            "upper": current_close,
            "index": n,
            "event_index": n,
            "type": "Range Deviation",
        }
        return build_signal(
            ctx,
            "SHORT",
            zone,
            "Range Deviation",
            "BSL/range-high sweep + close back inside range",
            sl_override=current_high * (1 + SL_BUFFER),
            tp_override=range_low,
        )

    return None


# ============================================================
# 12. MODEL RUNNER
# ============================================================

def run_models(
    df: pd.DataFrame,
    timeframe: str,
    higher_df: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Последовательно вызывает все модели; сбой одной модели не блокирует остальные."""
    signals: list[dict[str, Any]] = []
    try:
        ctx = analyze_smc(df)
    except Exception:
        logger.exception("Не удалось рассчитать SMC-контекст для %s", timeframe)
        return signals

    model_calls = []
    if timeframe == ENTRY_TIMEFRAME:
        model_calls.append(("Trend Continuation", lambda: model_continuation(df, higher_df, ctx)))
    model_calls.extend([
        ("Reversal via Sweep", lambda: model_reversal(df, ctx)),
        ("ICT Unicorn", lambda: model_unicorn(df, ctx)),
        ("Range Deviation", lambda: model_range(df, ctx)),
    ])

    for name, model_call in model_calls:
        try:
            signal = model_call()
            if signal is not None:
                signal["bar_time"] = df.iloc[-1]["timestamp"].strftime("%Y-%m-%d %H:%M UTC")
                signal["timeframe"] = timeframe
                signals.append(signal)
                logger.info("SETUP %s %s %s %s", timeframe, name, signal["direction"], signal["reason"])
        except Exception:
            logger.exception("Ошибка модели %s на %s", name, timeframe)

    return signals


# ============================================================
# 13. TELEGRAM
# ============================================================

def format_signal(symbol: str, timeframe: str, signal: dict[str, Any]) -> str:
    ticker = html.escape(symbol.split(":", 1)[-1])
    direction = signal["direction"]
    emoji = "🟢" if direction == "LONG" else "🔴"

    return (
        f"{emoji} <b>SMC / ICT SIGNAL</b>\n\n"
        f"<b>Symbol:</b> {ticker}\n"
        f"<b>Timeframe:</b> {html.escape(timeframe)}\n"
        f"<b>Model:</b> {html.escape(signal['model_name'])}\n"
        f"<b>Direction:</b> {direction}\n\n"
        f"<b>Entry:</b> {signal['entry']:.10g}\n"
        f"<b>Stop Loss:</b> {signal['sl']:.10g}\n"
        f"<b>Take Profit:</b> {signal['tp']:.10g}\n\n"
        f"<b>Price distance to SL:</b> {signal['price_risk_percent']:.3f}% of Entry\n"
        f"<b>Risk/Reward:</b> 1:{signal['rr']:.2f}\n"
        f"<b>Entry zone:</b> {html.escape(signal['zone'])}\n"
        + (
            f"\n<b>Position sizing (estimate):</b>\n"
            f"Risk budget: {signal['sizing']['risk_budget_usdt']:.2f} USDT "
            f"({signal['sizing']['risk_per_trade_pct']:.2f}% of equity)\n"
            f"Quantity: {signal['sizing']['quantity']:.6g} "
            f"{html.escape(ticker[:-4] if ticker.endswith('USDT') else ticker)}\n"
            f"Notional: {signal['sizing']['notional_usdt']:.2f} USDT "
            f"({signal['sizing']['equity_multiple']:.2f}x equity)\n"
            f"<i>Before fees, slippage and exchange quantity-step rounding.</i>\n"
            if signal.get("sizing") else ""
        )
        + f"\n<b>SMC confirmation:</b> {html.escape(signal['reason'])}\n"
        f"<b>Closed candle:</b> {html.escape(signal['bar_time'] or 'N/A')}"
    )


async def send_telegram_alert(
    session: aiohttp.ClientSession,
    symbol: str,
    timeframe: str,
    signal: dict[str, Any],
) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("Не заданы TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID")
        return

    signal_key = "|".join([
        symbol,
        timeframe,
        str(signal["model_name"]),
        str(signal["direction"]),
        str(signal["bar_time"]),
    ])
    if signal_was_sent(signal_key):
        logger.debug("Повторный сигнал пропущен: %s", signal_key)
        return

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": format_signal(symbol, timeframe, signal),
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:
        async with session.post(TELEGRAM_API_URL, json=payload) as response:
            result = await response.json(content_type=None)
            if response.status == 200 and result.get("ok"):
                remember_signal(signal_key)
                logger.info(
                    "Telegram: sent %s %s %s",
                    symbol,
                    timeframe,
                    signal["model_name"],
                )
            else:
                logger.error("Telegram API returned %s: %s", response.status, result)
    except Exception:
        logger.exception("Ошибка отправки Telegram для %s %s", symbol, timeframe)


# ============================================================
# 14. MARKET SCAN
# ============================================================

async def scan_market() -> None:
    timeout = aiohttp.ClientTimeout(total=25, connect=10)
    connector = aiohttp.TCPConnector(limit=20, ttl_dns_cache=300)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        try:
            symbols = await get_top_usdt_symbols(session)
        except Exception:
            logger.exception("Не удалось обновить universe Binance")
            if not _UNIVERSE_CACHE:
                return
            symbols = list(_UNIVERSE_CACHE)
            logger.warning("Используется последний кэшированный universe (%d тикеров)", len(symbols))

        semaphore = asyncio.Semaphore(MAX_TV_CONCURRENT_REQUESTS)
        jobs: list[tuple[str, str, asyncio.Task[pd.DataFrame | None]]] = []

        # Загружаем 1H и 4H по каждому инструменту с ограниченной параллельностью.
        for symbol in symbols:
            for timeframe in TIMEFRAMES:
                task = asyncio.create_task(fetch_data(symbol, timeframe, semaphore))
                jobs.append((symbol, timeframe, task))

        await asyncio.gather(*(task for _, _, task in jobs), return_exceptions=True)

        market_data: dict[tuple[str, str], pd.DataFrame | None] = {}
        for symbol, timeframe, task in jobs:
            if task.cancelled():
                market_data[(symbol, timeframe)] = None
                continue
            task_error = task.exception()
            if task_error is not None:
                logger.error("Fetch task failed %s %s: %s", symbol, timeframe, task_error)
                market_data[(symbol, timeframe)] = None
            else:
                market_data[(symbol, timeframe)] = task.result()

        signals_found = 0
        for symbol in symbols:
            df_1h = market_data.get((symbol, "1H"))
            df_4h = market_data.get((symbol, "4H"))

            # 1H — основной ТФ входа, 4H — контекст тренда.
            if df_1h is not None:
                signals = run_models(df_1h, "1H", higher_df=df_4h)
                for signal in signals:
                    signals_found += 1
                    await send_telegram_alert(session, symbol, "1H", signal)

            # На 4H отдельно ищем reversal / unicorn / range сетапы.
            # Trend Continuation на 4H не запускается, поскольку для него
            # не загружается дневной HTF-контекст.
            if df_4h is not None:
                signals = run_models(df_4h, "4H", higher_df=None)
                for signal in signals:
                    signals_found += 1
                    await send_telegram_alert(session, symbol, "4H", signal)

        logger.info(
            "Сканирование завершено: %d тикеров, %d OHLCV-серий, %d найденных сетапов",
            len(symbols),
            len(jobs),
            signals_found,
        )


# ============================================================
# 15. STARTUP / ONE-SHOT / CONTINUOUS MODES
# ============================================================

def validate_config() -> None:
    """Проверяет обязательные настройки перед запуском."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError("Добавь TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID в окружение или .env")
    if TOP_N_SYMBOLS < 1:
        raise RuntimeError("TOP_N_SYMBOLS должен быть >= 1")
    if MAX_TV_CONCURRENT_REQUESTS < 1:
        raise RuntimeError("MAX_TV_CONCURRENT_REQUESTS должен быть >= 1")
    if ACCOUNT_EQUITY_USDT < 0:
        raise RuntimeError("ACCOUNT_EQUITY_USDT должен быть >= 0")
    if RISK_PER_TRADE_PCT <= 0:
        raise RuntimeError("RISK_PER_TRADE_PCT должен быть > 0")


def log_startup() -> None:
    logger.info("SMC Scanner запущен; TOP_N=%d", TOP_N_SYMBOLS)
    logger.info("Universe filter: crypto assets only; bStocks/stablecoins/fiat/gold tokens excluded")
    if ACCOUNT_EQUITY_USDT > 0:
        logger.info(
            "Position sizing enabled: equity=%.2f USDT; risk/trade=%.2f%%",
            ACCOUNT_EQUITY_USDT,
            RISK_PER_TRADE_PCT,
        )
    else:
        logger.info("Position sizing disabled; set ACCOUNT_EQUITY_USDT to enable")
    logger.info("Входные ТФ: 1H и 4H; HTF-фильтр для 1H: 4H")


async def run_once() -> None:
    """Выполняет ровно один цикл сканирования (для GitHub Actions)."""
    validate_config()
    init_database()
    log_startup()
    logger.info("Режим one-shot: выполняется один цикл сканирования")
    await scan_market()
    logger.info("Однократное сканирование завершено")


async def main() -> None:
    """Непрерывный режим для локального ПК или постоянно работающего сервера."""
    validate_config()
    init_database()
    log_startup()
    logger.info("Непрерывный режим; интервал=%ds", SCAN_INTERVAL_SECONDS)

    while True:
        started = time.monotonic()
        try:
            await scan_market()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Ошибка основного цикла сканирования")

        elapsed = time.monotonic() - started
        sleep_for = max(1, SCAN_INTERVAL_SECONDS - elapsed)
        logger.info("Цикл занял %.1f сек.; следующий запуск через %.1f сек.", elapsed, sleep_for)
        await asyncio.sleep(sleep_for)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMC/ICT crypto market scanner")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Выполнить один проход сканирования и завершить работу (для GitHub Actions)",
    )
    args = parser.parse_args()

    try:
        asyncio.run(run_once() if args.once else main())
    except KeyboardInterrupt:
        logger.info("SMC Scanner остановлен пользователем")
