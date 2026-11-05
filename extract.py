"""Rule-based extraction of date, time, venue and audience from campus notices."""
import re
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

DATA = Path("data/notices.csv")
RESULTS = Path("results")
# The dataset was written relative to this date. The app itself uses date.today().
EVAL_REFERENCE_DATE = date(2026, 11, 3)

MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
WEEKDAYS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
            "friday": 4, "saturday": 5, "sunday": 6}
WEEKDAY_ABBR = {"mon": 0, "tues": 1, "tue": 1, "wed": 2, "thurs": 3, "thur": 3,
                "thu": 3, "fri": 4, "sat": 5, "sun": 6}
ANY_WEEKDAY = {**WEEKDAYS, **WEEKDAY_ABBR}

_MONTH = "|".join(sorted(MONTHS, key=len, reverse=True))
_WEEKDAY = "|".join(sorted(WEEKDAYS, key=len, reverse=True))
_WD_ANY = "|".join(sorted(ANY_WEEKDAY, key=len, reverse=True))
_ORD = r"(?:st|nd|rd|th)"

# "12th November 2026", "12 Nov"
DAY_MONTH = re.compile(
    rf"\b(\d{{1,2}}){_ORD}?(?:\s+of)?\s+({_MONTH})\b\.?(?:,?\s+(\d{{4}}))?", re.I)
# "November 12"
MONTH_DAY = re.compile(
    rf"\b({_MONTH})\s+(\d{{1,2}}){_ORD}?\b(?:,?\s+(\d{{4}}))?", re.I)
# "Thursday 12th" (no month given)
WD_ORD = re.compile(rf"\b({_WD_ANY})\.?,?\s+(\d{{1,2}}){_ORD}\b", re.I)
# "before 20th", "by 20th"
BY_ORD = re.compile(
    rf"\b(?:by|before|on|until|till)\s+(?:the\s+)?(\d{{1,2}}){_ORD}\b", re.I)
RELATIVE = re.compile(r"\b(today|tomorrow|tomorow|tmrw|tmr)\b", re.I)
# "this Friday", "on Monday"
WEEKDAY_RE = re.compile(rf"\b(?:(this|next|coming)\s+)?({_WEEKDAY})\b", re.I)
# used to ignore ranges such as "Monday to Friday"
RANGE_AFTER = re.compile(rf"\s*(?:to|-|–|and)\s*(?:{_WEEKDAY})\b", re.I)
RANGE_BEFORE = re.compile(rf"(?:{_WEEKDAY})\s*(?:to|-|–|and)\s*$", re.I)
TIME_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", re.I)


# ---------------------------------------------------------------- dates
def _safe_date(y, m, d):
    try:
        return date(y, m, d)
    except ValueError:
        return None


def _resolve_month_day(day, month, year, ref):
    if year:
        return _safe_date(year, month, day)
    d = _safe_date(ref.year, month, day)
    if d and d < ref - timedelta(days=30):  # a date well in the past means next year
        d = _safe_date(ref.year + 1, month, day)
    return d


def _next_weekday(wd, ref, force_next=False):
    ahead = (wd - ref.weekday()) % 7
    if force_next and ahead == 0:
        ahead = 7
    return ref + timedelta(days=ahead)


def _next_day_of_month(day, ref, weekday=None):
    y, m = ref.year, ref.month
    for _ in range(14):
        d = _safe_date(y, m, day)
        if d and d >= ref and (weekday is None or d.weekday() == weekday):
            return d
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return None


def _find_dates(text, ref):
    """Return [(position, date)] for every date expression, in text order."""
    found, spans = [], []

    def free(m):  # does this match overlap one we already used?
        return all(m.end() <= a or m.start() >= b for a, b in spans)

    def add(m, d):
        if d:
            found.append((m.start(), d))
            spans.append((m.start(), m.end()))

    for m in DAY_MONTH.finditer(text):
        year = int(m.group(3)) if m.group(3) else None
        add(m, _resolve_month_day(int(m.group(1)), MONTHS[m.group(2).lower()], year, ref))
    for m in MONTH_DAY.finditer(text):
        if free(m):
            year = int(m.group(3)) if m.group(3) else None
            add(m, _resolve_month_day(int(m.group(2)), MONTHS[m.group(1).lower()], year, ref))
    for m in WD_ORD.finditer(text):
        if free(m):
            add(m, _next_day_of_month(int(m.group(2)), ref, ANY_WEEKDAY[m.group(1).lower()]))
    for m in BY_ORD.finditer(text):
        if free(m):
            add(m, _next_day_of_month(int(m.group(1)), ref))
    for m in RELATIVE.finditer(text):
        if free(m):
            add(m, ref + timedelta(days=0 if m.group(1).lower() == "today" else 1))
    for m in WEEKDAY_RE.finditer(text):
        if not free(m):
            continue
        # "Thursday 12th November": the explicit date that follows wins over the weekday
        if any(m.end() <= a <= m.end() + 3 for a, _ in spans):
            continue
        if RANGE_AFTER.match(text, m.end()) or RANGE_BEFORE.search(text[:m.start()]):
            continue
        force_next = (m.group(1) or "").lower() == "next"
        add(m, _next_weekday(WEEKDAYS[m.group(2).lower()], ref, force_next))
    return sorted(found, key=lambda x: x[0])


def extract_date(text, ref=None):
    ref = ref or date.today()
    dates = _find_dates(text, ref)
    return dates[0][1] if dates else None   # simple rule: first date mentioned


def extract_time(text):
    m = TIME_RE.search(text)
    if m:
        hour = int(m.group(1)) % 12 + (12 if m.group(3).lower() == "pm" else 0)
        return f"{hour:02d}:{m.group(2) or '00'}"
    if re.search(r"\bmidnight\b", text, re.I):
        return "23:59"
    return None


# ---------------------------------------------------------------- venue
# Edit this list to match real Kirinyaga places. It is the "knowledge" of the extractor.
PLACES = [
    (r"dean of students(?: office)?", "Dean of Students office"),
    (r"dean['’]?s office", "Dean's office"),
    (r"registrar(?:['’]s)?(?: office)?", "Registrar's office"),
    (r"examinations? office", "Examinations Office"),
    (r"finance office", "Finance Office"),
    (r"career(?:s|['’]s)? office", "Career Office"),
    (r"attachment office", "Attachment Office"),
    (r"academic office", "Academic Office"),
    (r"security office", "Security office"),
    (r"chairman(?: of (?:the )?department)?(?:['’]s)?(?: office)?", "Chairman's office"),
    (r"ward administrator(?:['’]s)?(?: office)?", "Ward Administrator's office"),
    (r"your department(?: offices?)?", "Department offices"),
    (r"innovation hub", "Innovation Hub"),
    (r"student cent(?:re|er)", "Student Centre"),
    (r"main hall", "Main Hall"),
    (r"theatre hall", "Theatre Hall"),
    (r"main pitch", "Main pitch"),
    (r"main gate", "Main gate"),
    (r"university grounds", "University grounds"),
    (r"cafeteria", "Cafeteria"),
    (r"hostels?", "Hostels"),
    (r"library", "Library"),
    (r"student portal", "Student portal"),
    (r"e-?learning (?:platform|page)", "E-learning platform"),
    (r"helb portal", "HELB portal"),
    (r"jsm", "JSM"),
    (r"uppermess", "Uppermess"),
]


def extract_venue(text):
    hits = []  # (position, venue)
    for m in re.finditer(r"\b(computer\s+)?lab\s?(\d+)\b", text, re.I):
        hits.append((m.start(), f"{'Computer ' if m.group(1) else ''}Lab {m.group(2)}"))
    for m in re.finditer(r"\bLH\s?(\d+)\b", text, re.I):
        hits.append((m.start(), f"LH{m.group(1)}"))
    for pattern, name in PLACES:
        m = re.search(rf"\b(?:{pattern})\b", text, re.I)
        if m:
            hits.append((m.start(), name))
    if not hits:
        return None
    return min(hits, key=lambda h: (h[0], -len(h[1])))[1]   # earliest, then longest


# ------------------------------------------------------------- audience
def extract_audience(text):
    # Campus shorthand such as CSY2 = Computer Science, Year 2
    m = re.search(r"\bCSY(\d)\b", text, re.I)
    if m:
        return f"Year {m.group(1)} Computer Science"

    level = None
    m = re.search(r"\byear\s*(\d)(?:\s*(?:and|&|,|/)\s*(?:year\s*)?(\d))?\b", text, re.I)
    if m:
        level = f"Year {m.group(1)}" + (f" and {m.group(2)}" if m.group(2) else "")
    elif re.search(r"\bfinal[- ]year\b", text, re.I):
        level = "Final year"
    elif re.search(r"\bfirst[- ]years?\b", text, re.I):
        level = "Year 1"

    if level:
        programme = ""
        if re.search(r"computer science", text, re.I):
            programme = " Computer Science"
        elif re.search(r"\bIT\b", text):          # case-sensitive: avoids the word "it"
            programme = " IT"
        elif re.search(r"\bcomputing\b", text, re.I):
            programme = " computing"
        return level + programme

    groups = [
        (r"supplementary exams?", "Students with supplementary exams"),
        (r"graduation", "Final year students"),
        (r"instal?lment|overpaid", "Students with fee balances"),
        (r"women in tech|female", "Female students"),
        (r"continuing students", "Continuing students"),
        (r"graduating students|class of \d{4}", "Graduating students"),
        (r"fee balances?", "Students with fee balances"),
        (r"financial hardship|needy", "Students facing financial hardship"),
        (r"hostel", "Hostel residents"),
    ]
    for pattern, name in groups:
        if re.search(pattern, text, re.I):
            return name
    return "All students"


# ---------------------------------------------------------------- public API
def extract_all(text, ref=None):
    d = extract_date(text, ref)
    return {
        "date": d.isoformat() if d else None,
        "time": extract_time(text),
        "venue": extract_venue(text),
        "audience": extract_audience(text),
    }


# ---------------------------------------------------------------- evaluation
def _norm(s):
    return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()


def _match(pred, gt):
    """Lenient text match: equal, one contains the other, or >= 50% word overlap."""
    p, g = _norm(pred), _norm(gt)
    if not p or not g:
        return False
    if p == g or p in g or g in p:
        return True
    pw, gw = set(p.split()), set(g.split())
    return len(pw & gw) / len(pw | gw) >= 0.5


def evaluate(ref=EVAL_REFERENCE_DATE):
    RESULTS.mkdir(exist_ok=True)
    df = pd.read_csv(DATA).dropna(subset=["text"]).reset_index(drop=True)
    outputs = [extract_all(t, ref) for t in df["text"]]

    errors, summary = [], []
    for field in ["date", "venue", "audience"]:
        has_gt = df[field].notna()
        correct = 0
        for i in df.index[has_gt]:
            pred, gt = outputs[i][field], df.at[i, field]
            ok = (pred == gt) if field == "date" else _match(pred, gt)
            correct += bool(ok)
            if not ok:
                errors.append({"field": field, "text": df.at[i, "text"],
                               "expected": gt, "got": pred})
        # rows where the ground truth is blank but we still extracted something
        fp = None if field == "audience" else sum(
            1 for i in df.index[~has_gt] if outputs[i][field])
        n = int(has_gt.sum())
        summary.append({"field": field, "rows_with_truth": n, "correct": correct,
                        "accuracy": round(correct / n, 3) if n else None,
                        "false_positives": fp})

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(RESULTS / "extraction_summary.csv", index=False)
    pd.DataFrame(errors).to_csv(RESULTS / "extraction_errors.csv", index=False)
    print(summary_df.to_string(index=False))
    print(f"\n{len(errors)} errors saved to results/extraction_errors.csv")
    return summary_df


if __name__ == "__main__":
    demo = "Guys CAT for Data Structures is on Thursday 12th at 2pm, LH3. Don't miss."
    print(demo, "\n->", extract_all(demo, EVAL_REFERENCE_DATE), "\n")
    evaluate()
