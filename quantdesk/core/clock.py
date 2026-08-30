"""Time handling. Everything in this system is UTC, always, no exceptions.

The single most important convention in the codebase lives here:

    A bar is identified by its OPEN timestamp, but it only becomes KNOWABLE
    at its CLOSE timestamp.

Almost every lookahead bug in retail backtesting is a violation of that sentence:
a signal computed from a bar's close, then filled at that same bar's open (or
worse, at that close). `Bar.ts_close` exists so the engine can enforce it
mechanically rather than trusting the strategy author to remember.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

UTC = timezone.utc

# Regular US equity/index session: 09:30-16:00 ET = 6.5h = 23_400 seconds.
# ASSUMPTION (microstructure): we treat the RTH session as the unit of "one
# trading day" for annualisation. If you trade MNQ overnight (globex is ~23h),
# override `seconds_per_session` on the Timeframe or your Sharpe will be wrong
# by a factor of sqrt(23/6.5) ~= 1.88.
SECONDS_PER_RTH_SESSION = 23_400
SESSIONS_PER_YEAR = 252


def ensure_utc(ts: datetime) -> datetime:
    """Return `ts` as a timezone-aware UTC datetime.

    Naive datetimes are REJECTED rather than assumed-UTC. Silently localising a
    naive timestamp is how a whole dataset ends up shifted by the author's
    timezone offset, which then looks like a small alpha.
    """
    if ts.tzinfo is None:
        raise ValueError(
            f"naive datetime {ts!r}: timestamps must carry an explicit tzinfo. "
            "Parse with an explicit timezone rather than letting it default."
        )
    return ts.astimezone(UTC)


def utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0,
        second: int = 0) -> datetime:
    """Convenience constructor for UTC datetimes (mostly for tests/configs)."""
    return datetime(year, month, day, hour, minute, second, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class Timeframe:
    """A bar duration, plus the calendar facts needed to annualise from it.

    `periods_per_year` is deliberately not a magic constant. Annualising 5-minute
    bars with sqrt(252) instead of sqrt(19_656) understates volatility by ~8.8x,
    which makes any vol-targeted position size ~8.8x too large. That is a
    blow-up, not a rounding error, so the calendar is explicit and carried around
    with the timeframe.
    """

    label: str
    seconds: int
    seconds_per_session: int = SECONDS_PER_RTH_SESSION
    sessions_per_year: int = SESSIONS_PER_YEAR

    def __post_init__(self) -> None:
        if self.seconds <= 0:
            raise ValueError(f"timeframe seconds must be positive, got {self.seconds}")
        if self.seconds_per_session <= 0:
            raise ValueError("seconds_per_session must be positive")
        if self.sessions_per_year <= 0:
            raise ValueError("sessions_per_year must be positive")

    @property
    def delta(self) -> timedelta:
        return timedelta(seconds=self.seconds)

    @property
    def is_intraday(self) -> bool:
        return self.seconds < 86_400

    @property
    def periods_per_year(self) -> float:
        """Number of bars of this size in a trading year.

        Intraday: bars-per-session * sessions-per-year.
        Daily and slower: scaled off the session count, so daily -> 252.
        """
        if self.is_intraday:
            return self.sessions_per_year * (self.seconds_per_session / self.seconds)
        return self.sessions_per_year * (86_400 / self.seconds)

    @property
    def annualisation_factor(self) -> float:
        """sqrt(periods_per_year) - the multiplier from per-bar to annual vol."""
        return self.periods_per_year ** 0.5

    @classmethod
    def parse(cls, spec: str, **kwargs: int) -> "Timeframe":
        """Parse '1m', '5m', '15m', '1h', '4h', '1d', '1w' into a Timeframe."""
        text = spec.strip().lower()
        if len(text) < 2:
            raise ValueError(f"unparseable timeframe {spec!r}")
        unit, number = text[-1], text[:-1]
        if not number.isdigit():
            raise ValueError(f"unparseable timeframe {spec!r}")
        scale = {"s": 1, "m": 60, "h": 3_600, "d": 86_400, "w": 604_800}.get(unit)
        if scale is None:
            raise ValueError(f"unknown timeframe unit {unit!r} in {spec!r}")
        return cls(label=text, seconds=int(number) * scale, **kwargs)

    def close_of(self, ts_open: datetime) -> datetime:
        """The instant a bar opening at `ts_open` becomes knowable."""
        return ensure_utc(ts_open) + self.delta


MINUTE_1 = Timeframe("1m", 60)
MINUTE_5 = Timeframe("5m", 300)
MINUTE_15 = Timeframe("15m", 900)
HOUR_1 = Timeframe("1h", 3_600)
DAY_1 = Timeframe("1d", 86_400)
