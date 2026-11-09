"""Turn notices into a deadline table and an .ics calendar file."""
from datetime import date, datetime, timedelta

import pandas as pd

from src.extract import extract_all


def build_table(notices, today=None):
    """notices: list of dicts with 'text' and optional 'category'.
    Returns a DataFrame sorted by date (soonest first, undated last)."""
    today = today or date.today()
    rows = []
    for n in notices:
        info = extract_all(n["text"], today)
        d = date.fromisoformat(info["date"]) if info["date"] else None
        days_left = (d - today).days if d else None
        if d is None:
            status = "No date"
        elif days_left < 0:
            status = "Past"
        elif days_left <= 2:
            status = "Urgent"
        elif days_left <= 7:
            status = "This week"
        else:
            status = "Upcoming"
        rows.append({
            "category": n.get("category", ""),
            "date": d,
            "time": info["time"] or "",
            "days_left": days_left,
            "status": status,
            "venue": info["venue"] or "",
            "audience": info["audience"] or "",
            "text": n["text"],
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["_sort"] = df["date"].apply(lambda d: d or date.max)
    return df.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)


def _esc(s):
    return (str(s).replace("\\", "\\\\").replace(";", "\\;")
            .replace(",", "\\,").replace("\n", "\\n"))


def to_ics(table):
    """Build an .ics file from rows that have a date. Timed events last 1 hour."""
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0",
             "PRODID:-//CSK Campus Companion//EN", "CALSCALE:GREGORIAN"]
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    for i, r in table.dropna(subset=["date"]).iterrows():
        summary = f"[{r['category'] or 'Notice'}] {r['text'][:60]}"
        lines += ["BEGIN:VEVENT", f"UID:csk-{i}-{r['date']:%Y%m%d}@campus-companion",
                  f"DTSTAMP:{stamp}"]
        if r["time"]:
            start = datetime.strptime(f"{r['date']} {r['time']}", "%Y-%m-%d %H:%M")
            end = start + timedelta(hours=1)
            lines += [f"DTSTART:{start:%Y%m%dT%H%M%S}", f"DTEND:{end:%Y%m%dT%H%M%S}"]
        else:
            lines += [f"DTSTART;VALUE=DATE:{r['date']:%Y%m%d}",
                      f"DTEND;VALUE=DATE:{r['date'] + timedelta(days=1):%Y%m%d}"]
        lines += [f"SUMMARY:{_esc(summary)}", f"DESCRIPTION:{_esc(r['text'])}"]
        if r["venue"]:
            lines.append(f"LOCATION:{_esc(r['venue'])}")
        lines.append("END:VEVENT")
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines) + "\r\n"