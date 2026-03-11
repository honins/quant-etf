# Monitored ETF universe.
# Default pool is split into three classes:
# 1. Core tradable: broad-index, dividend, lower-volatility funds.
# 2. Satellite tradable: sector or thematic funds with higher beta.
# 3. Observe only: scored and reported, but excluded from the default trading pool.

CORE_TRADE_TICKERS = [
    "510050.SH",
    "510300.SH",
    "510500.SH",
    "512050.SH",
    "512100.SH",
    "588000.SH",
    "159915.SZ",
    "510880.SH",
    "512890.SH",
]

SATELLITE_TRADE_TICKERS = [
    "512880.SH",
    "512480.SH",
    "515030.SH",
    "518880.SH",
    "512400.SH",
    "515070.SH",
    "512170.SH",
    "512010.SH",
    "159992.SZ",
    "512800.SH",
    "159206.SZ",
    "159326.SZ",
    "159939.SZ",
    "512660.SH",
    "512690.SH",
]

OBSERVE_TICKERS = [
    "510900.SH",
    "513100.SH",
    "513330.SH",
    "513500.SH",
]

# Keep the most liquid code as the active representative for duplicated exposure.
DUPLICATE_TICKER_ALIASES = {
    "510330.SH": "510300.SH",
    "159919.SZ": "510300.SH",
    "512000.SH": "512880.SH",
}

WIDE_INDEX_TICKERS = CORE_TRADE_TICKERS.copy()
SECTOR_TICKERS = SATELLITE_TRADE_TICKERS.copy()
TRADABLE_TICKERS = CORE_TRADE_TICKERS + SATELLITE_TRADE_TICKERS
ACTIVE_TICKERS = TRADABLE_TICKERS + OBSERVE_TICKERS

CATEGORY_LABELS = {
    "core": "核心可交易",
    "satellite": "卫星可交易",
    "observe": "观察类",
    "unknown": "未分类",
}

ACTIVE_TICKER_NAMES = {
    "510050.SH": "上证50ETF",
    "510300.SH": "沪深300ETF",
    "510500.SH": "中证500ETF",
    "512050.SH": "A500ETF",
    "512100.SH": "中证1000ETF",
    "588000.SH": "科创50ETF",
    "159915.SZ": "创业板ETF",
    "510880.SH": "红利ETF",
    "512890.SH": "红利低波ETF",
    "512880.SH": "证券ETF",
    "512480.SH": "半导体ETF",
    "515030.SH": "新能源车ETF",
    "518880.SH": "黄金ETF",
    "512400.SH": "有色金属ETF",
    "515070.SH": "人工智能ETF",
    "512170.SH": "医疗ETF",
    "512010.SH": "医药ETF",
    "159992.SZ": "创新药ETF",
    "512800.SH": "银行ETF",
    "159206.SZ": "卫星ETF",
    "159326.SZ": "电网设备ETF",
    "159939.SZ": "信息技术ETF",
    "512660.SH": "军工ETF",
    "512690.SH": "酒ETF",
    "510900.SH": "H股ETF",
    "513100.SH": "纳指ETF",
    "513330.SH": "恒生互联网ETF",
    "513500.SH": "标普500ETF",
}

DUPLICATE_TICKER_NAMES = {
    "510330.SH": "沪深300ETF",
    "159919.SZ": "沪深300ETF",
    "512000.SH": "券商ETF",
}

TICKERS = ACTIVE_TICKER_NAMES | DUPLICATE_TICKER_NAMES


def normalize_ticker(code: str) -> str:
    return DUPLICATE_TICKER_ALIASES.get(code, code)


def is_duplicate_ticker(code: str) -> bool:
    return code in DUPLICATE_TICKER_ALIASES


def get_duplicate_aliases(code: str) -> list[str]:
    representative = normalize_ticker(code)
    return [alias for alias, target in DUPLICATE_TICKER_ALIASES.items() if target == representative]


def get_ticker_name(code: str) -> str:
    if code in TICKERS:
        return TICKERS[code]
    representative = normalize_ticker(code)
    return ACTIVE_TICKER_NAMES.get(representative, code)


def get_ticker_category(code: str) -> str:
    representative = normalize_ticker(code)
    if representative in CORE_TRADE_TICKERS:
        return "core"
    if representative in SATELLITE_TRADE_TICKERS:
        return "satellite"
    if representative in OBSERVE_TICKERS:
        return "observe"
    return "unknown"


def get_ticker_category_label(code: str) -> str:
    return CATEGORY_LABELS[get_ticker_category(code)]


def is_tradable_ticker(code: str) -> bool:
    return normalize_ticker(code) in TRADABLE_TICKERS


def get_tradable_ticker_list() -> list[str]:
    return TRADABLE_TICKERS.copy()


def get_observe_ticker_list() -> list[str]:
    return OBSERVE_TICKERS.copy()


def get_ticker_list(include_observe: bool = True) -> list[str]:
    tickers = TRADABLE_TICKERS.copy()
    if include_observe:
        tickers.extend(OBSERVE_TICKERS)
    return tickers
