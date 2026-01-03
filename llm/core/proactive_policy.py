from dataclasses import dataclass
from datetime import datetime, time
from typing import Optional

@dataclass
class ProactiveContext:
    user_detected: bool
    now_local: datetime
    last_spoke_at: Optional[datetime] = None
    quiet_hours: str = "23:00-07:00"  # can come from memory later

def _parse_quiet_hours(q: str):
    start_s, end_s = q.split("-")
    sh, sm = map(int, start_s.split(":"))
    eh, em = map(int, end_s.split(":"))
    return time(sh, sm), time(eh, em)

def _in_quiet_hours(now_t: time, q: str) -> bool:
    start, end = _parse_quiet_hours(q)
    if start < end:
        return start <= now_t < end
    # wraps midnight
    return now_t >= start or now_t < end

def should_start_conversation(ctx: ProactiveContext) -> bool:
    if not ctx.user_detected:
        return False
    if _in_quiet_hours(ctx.now_local.time(), ctx.quiet_hours):
        return False
    if ctx.last_spoke_at is None:
        return True
    # don't be annoying: at least 2 hours between proactive pings
    delta = ctx.now_local - ctx.last_spoke_at
    return delta.total_seconds() > 2 * 3600

def make_proactive_brief(now_local: datetime) -> str:
    # This is the *input* you give the model when it's time to speak.
    # Later you'll include calendar + vision signals.
    hour = now_local.hour
    if hour < 11:
        return "User just appeared. It's morning. Start a brief check-in and offer to prioritize the day."
    if hour < 18:
        return "User just appeared. It's daytime. Ask what they're heading into next and offer help."
    return "User just appeared. It's evening. Ask what needs to get done tonight and keep it tight."


