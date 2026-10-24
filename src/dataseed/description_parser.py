"""Deterministic plain-English -> schema parser (no LLM, no network).

HOW IT WORKS
    1. Find the record count (``5,000 records``), the domain (student, employee, sales,
       customer, fitness) and explicit column names (lists after "with", "including", ...).
    2. Give every column a default definition from a *name-based hint table* (``age`` -> integer
       18-65, ``gender`` -> Male/Female/Other, ``*_id`` -> identifier ...). Unknown names get a
       generic numeric default and a warning.
    3. Read rules that mention a column: ranges ("between 18 and 25", "at least 0"), allowed
       categories ("can be Basic, Silver or Gold"), uniqueness, missing values, mean/std,
       derived rules ("passed if marks are at least 40", "total = quantity * unit_price"),
       and correlations ("A and B are strongly positively correlated").
    4. If no columns are listed but a domain is recognised, use that domain's template.

LIMITATIONS (be aware!)
    * It is pattern matching, not language understanding. Unusual phrasing, negation, nested
      conditions, per-group rules or rules spanning sentences are NOT understood.
    * Sentences that look like rules but could not be parsed are reported under ``unsupported``.
    * Name-based defaults (ranges, categories) are *assumptions*; always review the schema.
    * Column names with spaces become snake_case (``pass status`` -> ``pass_status``).
"""
from __future__ import annotations

import re
from typing import Any

from .exceptions import ExpressionError, SchemaError
from .expressions import referenced_columns, parse_expression
from .models import DescriptionParseResult
from .schema import ColumnSpec, Schema

NUM = r"-?\d[\d,]*(?:\.\d+)?"
VAL = rf"(\d{{4}}-\d{{2}}-\d{{2}}|{NUM})"
GENERIC_TOKENS = {"id", "name", "type", "status", "date", "number", "value", "total", "the", "and", "data"}
STOP_ITEMS = {"records", "record", "rows", "row", "entries", "data", "dataset", "details", "information",
              "following", "the following", "samples", "fields", "columns", "it"}
STOP_WORDS = r"\b(?:must|should|between|can|may|is|are|from|where|at least|at most|than|ranging|if|when|will|has to)\b"
DOMAINS = {
    "student": ["student", "students", "school", "college", "university", "pupil"],
    "employee": ["employee", "employees", "staff", "workforce", "hr "],
    "sales": ["sales", "orders", "order", "transactions", "transaction", "retail"],
    "customer": ["customer", "customers", "client", "clients"],
    "fitness": ["fitness", "workout", "activity", "exercise", "health", "wearable"],
}
DEPARTMENTS = {
    "student": ["Computer Science", "Electronics", "Mechanical", "Civil", "Business"],
    "employee": ["Engineering", "Sales", "HR", "Finance", "Marketing", "Operations"],
}
CORR_STRENGTH = {"weakly": 0.3, "weak": 0.3, "moderately": 0.5, "moderate": 0.5, "strongly": 0.7,
                 "strong": 0.7, "highly": 0.8, "very strongly": 0.85}
OPS = [(r"greater than or equal to|at least|no less than|>=|≥", ">="), (r"less than or equal to|at most|no more than|<=|≤", "<="),
       (r"greater than|more than|above|over|exceeds", ">"), (r"less than|below|under", "<")]


def _n(s: str) -> float:
    return float(s.replace(",", ""))


def _num_or_int(v: float) -> int | float:
    return int(v) if float(v).is_integer() else v


def snake(name: str) -> str:
    s = re.sub(r"([a-z])([A-Z])", r"\1_\2", name.strip())
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _id_prefix(name: str) -> str:
    parts = [p for p in name.split("_") if p and p != "id"]
    return (parts[0][:3].upper() if parts else "ID") + "-"


def _hint(name: str, domain: str | None) -> tuple[dict, str] | None:
    """Name-based default definition for a column. Returns (spec-dict, explanation) or None."""
    t = set(name.split("_"))
    j = name.replace("_", "")

    def has(*words: str) -> bool:
        return bool(t & set(words)) or any(w in j for w in words if len(w) > 4)
    if name == "id" or name.endswith("_id") or t & {"uuid", "guid"}:
        return {"type": "identifier", "prefix": _id_prefix(name), "width": 6, "unique": True}, "identifier with sequential ids"
    if t & {"email", "mail"}:
        return {"type": "string", "text_kind": "email", "unique": True}, "synthetic e-mail addresses (example.com domains)"
    if t & {"phone", "mobile", "telephone"}:
        return {"type": "string", "text_kind": "phone"}, "synthetic phone numbers in the fictional 555-01XX range"
    if "address" in t:
        return {"type": "string", "text_kind": "address"}, "synthetic street address"
    if t & {"company", "employer", "organization", "organisation"}:
        return {"type": "string", "text_kind": "company"}, "synthetic company name"
    if "name" in t and not t & {"department", "product", "category", "city"}:
        kind = "first_name" if "first" in t else "last_name" if t & {"last", "surname"} else "name"
        return {"type": "string", "text_kind": kind}, f"synthetic {kind.replace('_', ' ')}s"
    if name in ("age",) or name.endswith("_age"):
        lo, hi = (18, 25) if domain == "student" else (18, 65)
        return {"type": "integer", "min": lo, "max": hi}, f"integer age {lo}-{hi}"
    if t & {"gender", "sex"}:
        return {"type": "categorical", "categories": ["Male", "Female", "Other"], "probabilities": [0.48, 0.48, 0.04]}, "Male/Female/Other (48/48/4%)"
    if "department" in t or "dept" in t:
        cats = DEPARTMENTS.get(domain or "", DEPARTMENTS["employee"])
        return {"type": "categorical", "categories": cats}, f"departments {cats} with equal probability"
    if t & {"attendance", "percentage", "percent", "pct"}:
        return {"type": "float", "min": 0, "max": 100, "distribution": "normal", "mean": 75, "std": 15, "decimals": 1}, "percentage 0-100 (normal, mean 75, sd 15)"
    if t & {"marks", "score", "scores", "gpa"} and "gpa" not in t:
        return {"type": "float", "min": 0, "max": 100, "distribution": "normal", "mean": 65, "std": 15, "decimals": 1}, "score 0-100 (normal, mean 65, sd 15)"
    if "gpa" in t:
        return {"type": "float", "min": 0, "max": 10, "distribution": "normal", "mean": 7, "std": 1.2, "decimals": 2}, "GPA 0-10"
    if t & {"salary", "income", "wage"}:
        return {"type": "integer", "min": 20000, "max": 200000, "distribution": "lognormal", "mean": 65000, "std": 25000}, "salary 20,000-200,000 (lognormal)"
    if t & {"price", "cost"} or name == "unit_price":
        return {"type": "float", "min": 1, "max": 1000, "distribution": "lognormal", "mean": 60, "std": 80, "decimals": 2}, "price 1-1000 (lognormal)"
    if t & {"amount", "spending", "revenue", "sales", "total", "spend", "balance"}:
        return {"type": "float", "min": 0, "max": 20000, "distribution": "lognormal", "mean": 1500, "std": 1500, "decimals": 2}, "amount 0-20,000 (lognormal)"
    if t & {"quantity", "qty", "count"}:
        return {"type": "integer", "min": 1, "max": 20}, "integer 1-20"
    if "rating" in t:
        return {"type": "integer", "min": 1, "max": 5}, "integer rating 1-5"
    if t & {"experience", "tenure"} or (("years" in t) and not t & {"birth"}):
        return {"type": "integer", "min": 0, "max": 35, "distribution": "normal", "mean": 8, "std": 6}, "years 0-35"
    if "weight" in t:
        return {"type": "float", "min": 35, "max": 150, "distribution": "normal", "mean": 70, "std": 13, "decimals": 1}, "weight 35-150 kg"
    if "height" in t:
        return {"type": "float", "min": 130, "max": 210, "distribution": "normal", "mean": 170, "std": 10, "decimals": 1}, "height 130-210 cm"
    if "steps" in t:
        return {"type": "integer", "min": 0, "max": 30000, "distribution": "normal", "mean": 8000, "std": 3500}, "steps 0-30,000"
    if "calories" in t:
        return {"type": "integer", "min": 800, "max": 5000, "distribution": "normal", "mean": 2200, "std": 500}, "calories 800-5000"
    if "heart" in t:
        return {"type": "integer", "min": 40, "max": 120, "distribution": "normal", "mean": 68, "std": 9}, "heart rate 40-120"
    if "sleep" in t:
        return {"type": "float", "min": 3, "max": 11, "distribution": "normal", "mean": 7, "std": 1.1, "decimals": 1}, "sleep hours 3-11"
    if t & {"minutes", "duration"}:
        return {"type": "integer", "min": 5, "max": 180}, "minutes 5-180"
    if "dob" in t or "birth" in t:
        return {"type": "date", "min": "1960-01-01", "max": "2005-12-31"}, "birth date 1960-2005"
    if t & {"date", "joined", "joining", "signup", "created", "timestamp"} or name.endswith("_at"):
        return {"type": "date", "min": "2018-01-01", "max": "2025-12-31"}, "date 2018-01-01..2025-12-31"
    if "status" in t and t & {"pass", "result"}:
        return {"type": "categorical", "categories": ["Pass", "Fail"]}, "Pass/Fail"
    if t & {"passed", "pass"} or t & {"active", "enrolled", "flag"} or name.startswith(("is_", "has_")):
        return {"type": "boolean", "probability": 0.5}, "boolean, 50% True"
    if "status" in t:
        return {"type": "categorical", "categories": ["Active", "Inactive"]}, "Active/Inactive"
    if "grade" in t:
        return {"type": "categorical", "categories": ["A", "B", "C", "D", "F"]}, "grades A-F"
    if t & {"payment", "method"}:
        return {"type": "categorical", "categories": ["Card", "Cash", "UPI", "Bank Transfer"]}, "payment methods"
    if "region" in t:
        return {"type": "categorical", "categories": ["North", "South", "East", "West"]}, "regions"
    if "city" in t:
        return {"type": "string", "text_kind": "city"}, "synthetic city names"
    if t & {"category", "product"}:
        return {"type": "categorical", "categories": ["Electronics", "Clothing", "Home", "Grocery", "Sports"]}, "product categories"
    if t & {"job", "title", "role", "designation"}:
        return {"type": "categorical", "categories": ["Analyst", "Engineer", "Manager", "Associate", "Director"]}, "job titles"
    if t & {"membership", "tier", "segment", "level", "type", "plan"}:
        return {"type": "categorical", "categories": ["A", "B", "C"], "_placeholder": True}, "PLACEHOLDER categories A/B/C"
    return None


TEMPLATES: dict[str, dict] = {
    "student": {"columns": ["student_id", "age", "gender", "department", "attendance_percentage", "marks", "passed"],
                "rules": {"passed": "marks >= 40"}, "corr": [("attendance_percentage", "marks", 0.5)]},
    "employee": {"columns": ["employee_id", "age", "gender", "department", "years_experience", "salary", "joining_date", "is_active"],
                 "rules": {}, "corr": [("years_experience", "salary", 0.6)]},
    "sales": {"columns": ["order_id", "order_date", "product_category", "quantity", "unit_price", "total_amount", "payment_method", "region"],
              "rules": {"total_amount": "quantity * unit_price"}, "corr": []},
    "customer": {"columns": ["customer_id", "age", "gender", "city", "membership_type", "annual_spending", "signup_date"],
                 "rules": {}, "corr": []},
    "fitness": {"columns": ["user_id", "age", "gender", "weight_kg", "height_cm", "daily_steps", "calories_burned",
                            "workout_minutes", "resting_heart_rate", "sleep_hours"],
                "rules": {}, "corr": [("daily_steps", "calories_burned", 0.6), ("weight_kg", "height_cm", 0.4)]},
}


class _Parser:
    def __init__(self, text: str):
        self.text = " ".join(text.strip().split())
        self.res = DescriptionParseResult(schema=None)
        self.consumed: set[int] = set()
        self.sent_spans = [(m.start(), m.end()) for m in re.finditer(r"[^.!?]+(?:[.!?](?!\d)|$)", self.text)] or [(0, len(self.text))]
        self.cols: dict[str, dict] = {}
        self.relationships: list[dict] = []
        self.constraints: list[dict] = []
        self.rules: dict[str, str] = {}
        self.domain: str | None = None
        self.placeholders: set[str] = set()

    # -------------------------------------------------------------- helpers
    def mark(self, pos: int) -> None:
        for i, (a, b) in enumerate(self.sent_spans):
            if a <= pos < b + 1:
                self.consumed.add(i)

    def understood(self, msg: str, pos: int | None = None) -> None:
        self.res.understood.append(msg)
        if pos is not None:
            self.mark(pos)

    def assume(self, msg: str) -> None:
        self.res.assumptions.append(msg)

    # -------------------------------------------------------------- extraction
    def find_rows(self) -> None:
        m = re.search(rf"(\d[\d,]*)\s*(k|thousand|million)?\s+(?:\w+\s+){{0,2}}?(?:records?|rows?|entries|samples?|observations?|"
                      r"students?|employees?|customers?|orders?|transactions?|people|persons?|patients?|users?|items?|products?)\b",
                      self.text, re.I)
        if not m:
            return
        n = _n(m.group(1)) * {"k": 1e3, "thousand": 1e3, "million": 1e6}.get((m.group(2) or "").lower(), 1)
        if n >= 1 and float(n).is_integer():
            self.res.n_rows = int(n)
            self.understood(f"Record count: {int(n):,}", m.start())

    def find_domain(self) -> None:
        low = " " + self.text.lower() + " "
        best = None
        for dom, words in DOMAINS.items():
            for w in words:
                i = low.find(" " + w.strip() if not w.endswith(" ") else " " + w)
                if i >= 0 and (best is None or i < best[0]):
                    best = (i, dom)
        if best:
            self.domain = self.res.domain = best[1]
            self.understood(f"Domain recognised: {best[1]}")

    @staticmethod
    def _clean_item(item: str) -> tuple[str | None, bool]:
        """Return (column name or None, stop) for one list item."""
        item = item.strip(" \t\"'`()")
        cut = re.search(STOP_WORDS, item, re.I)
        stop = False
        if cut:
            item, stop = item[:cut.start()].strip(), True
        item = re.sub(r"^(?:and|or|a|an|the|their|its|also|plus)\s+", "", item, flags=re.I)
        item = re.sub(r"\s+(?:columns?|fields?|attributes?)$", "", item, flags=re.I).strip()
        if not item or item.lower() in STOP_ITEMS or not re.match(r"^[A-Za-z][A-Za-z0-9_\- ]*$", item) or len(item.split()) > 4:
            return None, stop
        return snake(item), stop

    def find_columns(self) -> list[str]:
        trig = re.compile(r"\b(?:with|including|includes|include|contains?|containing|columns?(?:\s+(?:are|for|named|called))?|"
                          r"fields?(?:\s+(?:are|for|named|called))?|having|has|have|featuring|attributes?)\b\s*:?\s*", re.I)
        names: list[str] = []
        for si, (a, b) in enumerate(self.sent_spans):
            sent = self.text[a:b]
            ms = list(trig.finditer(sent))
            for k, m in enumerate(ms):
                end = ms[k + 1].start() if k + 1 < len(ms) else len(sent)
                seg = re.split(r"[.;]|\bwhere\b|\bsuch that\b|\bso that\b", sent[m.end():end], maxsplit=1)[0]
                found = []
                for item in re.split(r",|\band\b|&|\bor\b|/", seg):
                    name, stop = self._clean_item(item)
                    if name:
                        found.append(name)
                    if stop:
                        break
                if found:
                    names += [f for f in found if f not in names]
                    self.consumed.add(si)
        return names

    # -------------------------------------------------------------- building
    def add_column(self, name: str) -> None:
        h = _hint(name, self.domain)
        if h:
            spec, why = h
            spec = dict(spec)
            if spec.pop("_placeholder", False):
                self.placeholders.add(name)
            self.assume(f"Column '{name}': {why} (default from the column name; override by stating a rule).")
        else:
            spec = {"type": "float", "min": 0, "max": 100}
            self.res.warnings.append(f"Column '{name}': type could not be inferred from the name; assumed a number between 0 and 100.")
        spec["name"] = name
        spec["source"] = "parsed"
        self.cols[name] = spec

    def alias_map(self) -> dict[str, str]:
        amap: dict[str, str] = {}
        tok_owner: dict[str, set[str]] = {}
        for name in self.cols:
            for t in name.split("_"):
                if len(t) > 3 and t not in GENERIC_TOKENS:
                    tok_owner.setdefault(t, set()).add(name)
        for name in self.cols:
            amap[name.lower()] = name
            amap[name.replace("_", " ").lower()] = name
        for t, owners in tok_owner.items():
            if len(owners) == 1:
                amap.setdefault(t, next(iter(owners)))
        return amap

    def alias_re(self, amap: dict[str, str]) -> re.Pattern:
        alts = sorted(amap, key=len, reverse=True)
        return re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(re.escape(a) for a in alts) + r")(?![A-Za-z0-9_])", re.I)

    def apply_constraints(self) -> None:
        if not self.cols:
            return
        amap = self.alias_map()
        rx = self.alias_re(amap)
        matches = list(rx.finditer(self.text))
        fill = r"^\s*,?\s*(?:(?:must|should|has to|have to|needs? to|can|may|will|shall|is|are|to)\s+)*(?:be\s+)?"
        for i, m in enumerate(matches):
            col = amap[m.group(1).lower()]
            end = len(self.text)
            sent_end = re.search(r"[.;!?](?:\s|$)", self.text[m.end():])
            if sent_end:
                end = m.end() + sent_end.start()
            if i + 1 < len(matches):
                end = min(end, matches[i + 1].start())
            sent_start = max([a for a, b in self.sent_spans if a <= m.start()] or [0])
            if re.search(r"\b(?:if|when|unless)\b", self.text[sent_start:m.start()], re.I):
                continue   # the mention is inside a condition (handled by the rule parsers)
            w = self.text[m.end():end]
            self._column_window(col, w, fill, m.start())

    def _column_window(self, col: str, w: str, fill: str, pos: int) -> None:
        spec = self.cols[col]
        r = re.match(fill + rf"(?:between|from|ranging from|in the range(?: of)?|within)\s+{VAL}\s*(?:and|to|-|–)\s*{VAL}", w, re.I)
        if r:
            self._set_range(col, r.group(1), r.group(2), pos)
            return
        cat_list = re.match(fill + r"(?:one of|either|any of|take(?: the values?)?)?\s*(.+)$", w, re.I) if re.match(
            r"^\s*,?\s*(?:(?:can|may|must|should|will)\s+)?(?:be|take)\b", w, re.I) else None
        for pat, op in OPS:
            m = re.match(fill + rf"(?:{pat})\s*{VAL}", w, re.I)
            if m:
                self._set_bound(col, op, m.group(1), pos)
                return
        m = re.match(r"^\s*,?\s*(?:(?:must|should|will|can|may)\s+)*(?:be\s+)?(?:minimum|min)(?:imum)?(?: of)?\s*" + VAL, w, re.I)
        if m:
            self._set_bound(col, ">=", m.group(1), pos)
            return
        m = re.match(r"^\s*,?\s*(?:(?:must|should|will|can|may)\s+)*(?:be\s+)?(?:maximum|max)(?:imum)?(?: of)?\s*" + VAL, w, re.I)
        if m:
            self._set_bound(col, "<=", m.group(1), pos)
            return
        if re.match(r"^\s*(?:must|should|needs? to|has to|is|are)\s+(?:be\s+)?unique", w, re.I):
            spec["unique"] = True
            self.understood(f"{col} must be unique", pos)
            return
        m = re.match(r"^\s*(?:can|may|could)\s+(?:be\s+)?(?:missing|null|empty|blank)", w, re.I)
        if m:
            spec["missing_rate"], spec["nullable"] = 0.05, True
            self.assume(f"Column '{col}': allowed to be missing but no rate given; assumed 5%.")
            self.understood(f"{col} may be missing", pos)
            return
        m = re.match(r"^\s*(?:has|with|contains)\s+(?:about\s+)?(\d+(?:\.\d+)?)\s*%\s*(?:of\s+)?(?:its\s+)?(?:values\s+)?(?:missing|null|empty|blank)", w, re.I)
        if m:
            self._set_missing(col, float(m.group(1)), pos)
            return
        m = re.search(r"(?:normal(?:ly)? distributed|follows? a normal distribution)", w, re.I)
        if m:
            self._set_normal(col, w, pos)
            return
        m = re.match(r"^\s*(?:is|are)\s+(?:uniformly distributed|uniform)", w, re.I)
        if m and spec["type"] in ("integer", "float") and spec.get("min") is not None:
            spec["distribution"] = "uniform"
            for k in ("mean", "std"):
                spec.pop(k, None)
            self.understood(f"{col} uniformly distributed", pos)
            return
        if cat_list:
            self._set_categories(col, cat_list.group(1), pos)

    def _set_range(self, col: str, a: str, b: str, pos: int) -> None:
        spec = self.cols[col]
        if spec["type"] in ("date", "datetime") or (re.match(r"\d{4}-", a) and re.match(r"\d{4}-", b)):
            if not (re.match(r"\d{4}-", a) and re.match(r"\d{4}-", b)):
                self.res.warnings.append(f"Column '{col}': date range needs ISO dates (YYYY-MM-DD); ignored.")
                return
            spec["type"] = spec["type"] if spec["type"] in ("date", "datetime") else "date"
            spec["min"], spec["max"] = a, b
            for k in ("probability", "categories", "text_kind", "decimals"):
                spec.pop(k, None)
            self.understood(f"{col} between {a} and {b}", pos)
            return
        if spec["type"] in ("categorical", "boolean", "identifier", "string"):
            self.res.unsupported.append(f"Range for '{col}' ignored: column is '{spec['type']}' (name-based default). "
                                        "Use a different column name or edit the schema.")
            return
        lo, hi = _n(a), _n(b)
        if lo > hi:
            self.res.warnings.append(f"Column '{col}': range {a}..{b} is reversed; swapped.")
            lo, hi = hi, lo
        self._apply_numeric_range(col, lo, hi)
        self.understood(f"{col} between {_num_or_int(lo)} and {_num_or_int(hi)}", pos)

    def _apply_numeric_range(self, col: str, lo: float | None, hi: float | None) -> None:
        spec = self.cols[col]
        lo = _num_or_int(lo) if lo is not None else spec.get("min")
        hi = _num_or_int(hi) if hi is not None else spec.get("max")
        if lo is None or hi is None:
            self.res.warnings.append(f"Column '{col}': only one bound is known.")
        spec["min"], spec["max"] = lo, hi
        d = spec.get("distribution")
        if d in ("normal", "lognormal"):
            mean = spec.get("mean")
            if mean is None or lo is None or hi is None or not lo < mean < hi:
                spec["distribution"] = "normal"
                spec["mean"], spec["std"] = (lo + hi) / 2, (hi - lo) / 6
                self.assume(f"Column '{col}': mean/std not stated; assumed mean={spec['mean']:g}, std={spec['std']:g} within the range.")
        elif d is None:
            spec["distribution"] = "uniform"
            self.assume(f"Column '{col}': no distribution stated; values are uniform within the range.")

    def _set_bound(self, col: str, op: str, v: str, pos: int) -> None:
        spec = self.cols[col]
        if spec["type"] not in ("integer", "float", "date", "datetime"):
            self.res.unsupported.append(f"Bound '{op} {v}' for '{col}' ignored: column is '{spec['type']}'.")
            return
        x = v if re.match(r"\d{4}-", v) else _n(v)
        if isinstance(x, str):
            spec["min" if op in (">", ">=") else "max"] = x
            self.understood(f"{col} {op} {v}", pos)
            return
        if op in (">", "<") and spec["type"] == "integer":
            x = x + 1 if op == ">" else x - 1
        elif op in (">", "<"):
            self.constraints.append({"type": "expression", "expression": f"{col} {op} {_num_or_int(x)}"})
            self.res.warnings.append(f"Column '{col}': strict '{op}' on a decimal column is enforced as a row constraint, not as a distribution bound.")
        if op in (">", ">="):
            self._apply_numeric_range(col, x, None)
        else:
            self._apply_numeric_range(col, None, x)
        self.understood(f"{col} {op} {v}", pos)

    def _set_categories(self, col: str, lst: str, pos: int) -> None:
        items = [re.sub(r"^(?:and|or)\s+", "", i.strip(" \t\"'."), flags=re.I) for i in re.split(r",|\bor\b|\band\b|/", lst)]
        items = [i for i in items if i]
        bad = re.search(STOP_WORDS + r"|unique|missing|null|distributed", lst, re.I)
        if bad:   # list continues into another clause; keep only items before it
            items = [re.sub(r"^(?:and|or)\s+", "", i.strip(" \t\"'."), flags=re.I) for i in re.split(r",|\bor\b|\band\b|/", lst[:bad.start()])]
            items = [i for i in items if i]
        if len(items) < 2:
            return
        spec = self.cols[col]
        numeric = all(re.fullmatch(NUM, i) for i in items)
        if numeric:
            vals = [_num_or_int(_n(i)) for i in items]
            spec.update(type="integer" if all(isinstance(v, int) for v in vals) else "float", categories=vals)
            for k in ("min", "max", "distribution", "mean", "std", "decimals", "probability", "text_kind", "prefix", "width"):
                spec.pop(k, None)
        else:
            spec.clear()
            spec.update(name=col, type="categorical", categories=list(dict.fromkeys(items)), source="parsed")
        self.placeholders.discard(col)
        self.understood(f"{col} can be one of {items}", pos)
        self.assume(f"Column '{col}': categories have equal probability (none were stated).")

    def _set_missing(self, col: str, pct: float, pos: int) -> None:
        spec = self.cols[col]
        if not 0 <= pct < 100:
            self.res.warnings.append(f"Column '{col}': missing percentage {pct} is not in [0, 100).")
            return
        spec["missing_rate"], spec["nullable"] = round(pct / 100, 6), pct > 0
        self.understood(f"{col}: {pct:g}% missing values", pos)

    def _set_normal(self, col: str, w: str, pos: int) -> None:
        spec = self.cols[col]
        if spec["type"] not in ("integer", "float"):
            return
        mean = re.search(r"(?:mean|average)\s*(?:of|=|is)?\s*(" + NUM + ")", w, re.I)
        std = re.search(r"(?:standard deviation|std|sd)\s*(?:of|=|is)?\s*(" + NUM + ")", w, re.I)
        spec["distribution"] = "normal"
        if mean and std:
            spec["mean"], spec["std"] = _n(mean.group(1)), _n(std.group(1))
            self.understood(f"{col} ~ normal(mean={spec['mean']:g}, std={spec['std']:g})", pos)
        else:
            self.res.warnings.append(f"Column '{col}': 'normal' needs both mean and standard deviation; existing defaults kept.")
            if spec.get("mean") is None and spec.get("min") is not None and spec.get("max") is not None:
                spec["mean"], spec["std"] = (spec["min"] + spec["max"]) / 2, (spec["max"] - spec["min"]) / 6

    # -------------------------------------------------------------- global rules
    def global_rules(self) -> None:
        if not self.cols:
            return
        amap = self.alias_map()
        arx = self.alias_re(amap).pattern
        # no missing values
        m = re.search(r"\b(?:no|without|zero)\s+missing(?:\s+values?)?|\bno\s+(?:null|blank|empty)\s+values?|\bwithout\s+(?:any\s+)?(?:nulls|blanks)", self.text, re.I)
        if m:
            for s in self.cols.values():
                s["missing_rate"], s["nullable"] = 0.0, False
            self.understood("No missing values in any column", m.start())
        # N% of col values are missing
        for m in re.finditer(rf"(\d+(?:\.\d+)?)\s*%\s*(?:of\s+)?(?:the\s+)?{arx}\s*(?:values?\s*)?(?:are|is|should be|to be)?\s*(?:missing|null|blank|empty)", self.text, re.I):
            self._set_missing(amap[m.group(2).lower()], float(m.group(1)), m.start())
        # pass rule
        opx = "|".join(p for p, _ in OPS)
        for m in re.finditer(rf"(?:pass(?:es|ed)?|qualif(?:y|ies|ied))\s+(?:if|when)\s+(?:their\s+|his\s+|her\s+|the\s+)?{arx}\s+(?:is|are)?\s*({opx})\s*({NUM})", self.text, re.I):
            src, cond = amap[m.group(1).lower()], self._op(m.group(2))
            tgt = next((c for c in self.cols if set(c.split("_")) & {"pass", "passed", "result"}), None)
            if tgt is None:
                tgt = "passed"
                self.add_column(tgt)
                self.res.warnings.append("A pass rule was found but no pass/result column was listed; added boolean column 'passed'.")
            self.rules[tgt] = f"{src} {cond} {_num_or_int(_n(m.group(3)))}"
            self.understood(f"{tgt} derived: {self.rules[tgt]}", m.start())
        # "X is true if Y op N"
        for m in re.finditer(rf"{arx}\s+(?:is|=)\s+(?:true|yes)\s+(?:if|when)\s+{arx}\s*(?:is|are)?\s*({opx})\s*({NUM})", self.text, re.I):
            tgt, src = amap[m.group(1).lower()], amap[m.group(2).lower()]
            self.rules[tgt] = f"{src} {self._op(m.group(3))} {_num_or_int(_n(m.group(4)))}"
            self.cols[tgt]["type"] = "boolean"
            self.understood(f"{tgt} derived: {self.rules[tgt]}", m.start())
        # arithmetic definitions
        for m in re.finditer(rf"{arx}\s*(?:=|equals|is equal to|is calculated as|is computed as)\s*([^.;]+)", self.text, re.I):
            tgt = amap[m.group(1).lower()]
            expr = self._rhs(m.group(2), amap)
            if expr:
                self.rules[tgt] = expr
                self.understood(f"{tgt} derived: {expr}", m.start())
            else:
                self.res.unsupported.append(f"Could not interpret definition of '{tgt}': {m.group(0).strip()!r}")
                self.mark(m.start())
        for m in re.finditer(rf"{arx}\s+is\s+the\s+(product|sum)\s+of\s+{arx}\s+and\s+{arx}", self.text, re.I):
            tgt, a, b = (amap[m.group(i).lower()] for i in (1, 3, 4))
            self.rules[tgt] = f"{a} {'*' if m.group(2).lower() == 'product' else '+'} {b}"
            self.understood(f"{tgt} derived: {self.rules[tgt]}", m.start())
        # correlations
        for m in re.finditer(rf"{arx}\s+and\s+{arx}\s+(?:are|should be|must be)\s+(?:(very strongly|strongly|moderately|weakly|highly)\s+)?(?:(positively|negatively)\s+)?correlated(?:\s+(?:with\s+)?(?:a\s+)?(?:correlation\s+)?(?:of\s+)?(-?0?\.\d+))?", self.text, re.I):
            a, b = amap[m.group(1).lower()], amap[m.group(2).lower()]
            self._corr(a, b, m.group(3), m.group(4), m.group(5), m.start())
        for m in re.finditer(rf"correlation\s+(?:of|=|coefficient of)\s*(-?0?\.\d+)\s+between\s+{arx}\s+and\s+{arx}", self.text, re.I):
            self._corr(amap[m.group(2).lower()], amap[m.group(3).lower()], None, None, m.group(1), m.start())
        for m in re.finditer(rf"{arx}\s+(?:is|are)\s+(?:(strongly|moderately|weakly)\s+)?(positively|negatively)\s+correlated\s+with\s+{arx}", self.text, re.I):
            self._corr(amap[m.group(1).lower()], amap[m.group(4).lower()], m.group(2), m.group(3), None, m.start())

    @staticmethod
    def _op(phrase: str) -> str:
        p = phrase.lower()
        for pat, op in OPS:
            if re.fullmatch(pat, p):
                return op
        return ">="

    def _corr(self, a: str, b: str, strength, sign, value, pos: int) -> None:
        if a == b:
            return
        if value is not None:
            v = float(value)
        else:
            v = CORR_STRENGTH.get((strength or "moderately").lower(), 0.5)
            if strength is None:
                self.assume(f"Correlation between {a} and {b}: strength not stated; assumed 0.5.")
            if (sign or "positively").lower() == "negatively":
                v = -v
        self.relationships.append({"type": "correlation", "columns": [a, b], "value": v, "source": "user"})
        self.understood(f"{a} and {b} correlated at {v:+.2f}", pos)

    def _rhs(self, rhs: str, amap: dict[str, str]) -> str | None:
        rhs = rhs.strip().rstrip(",")
        rhs = re.sub(r"\s+(?:and|where|if|when|because)\s+.*$", "", rhs, flags=re.I) if not re.search(r"\band\b.*\b(?:and)\b", rhs) else rhs
        for pat, sym in ((r"multiplied by|times|\bx\b", "*"), (r"divided by", "/"), (r"\bplus\b", "+"), (r"\bminus\b", "-")):
            rhs = re.sub(pat, f" {sym} ", rhs, flags=re.I)
        for alias in sorted(amap, key=len, reverse=True):
            rhs = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(alias)}(?![A-Za-z0-9_])", amap[alias], rhs, flags=re.I)
        rhs = rhs.strip()
        try:
            refs = referenced_columns(rhs)
            parse_expression(rhs)
        except ExpressionError:
            return None
        return rhs if refs and all(r in self.cols for r in refs) else None

    # -------------------------------------------------------------- assemble
    def build(self) -> DescriptionParseResult:
        res = self.res
        if not self.text:
            res.warnings.append("The description is empty.")
            return res
        self.find_rows()
        self.find_domain()
        names = self.find_columns()
        tmpl = TEMPLATES.get(self.domain or "")
        if names:
            self.understood(f"Columns listed: {names}")
        elif tmpl:
            names = list(tmpl["columns"])
            self.assume(f"No columns were listed; used the '{self.domain}' template: {names}.")
            self.rules.update({k: v for k, v in tmpl["rules"].items()})
            for a, b, v in tmpl["corr"]:
                self.relationships.append({"type": "correlation", "columns": [a, b], "value": v, "source": "assumed"})
                self.assume(f"Template correlation {a}~{b} = {v} (typical, not data-driven).")
        else:
            res.warnings.append("No column names could be identified and no known domain (student, employee, sales, "
                                "customer, fitness) was mentioned. List the columns, e.g. 'with id, age, city and salary'.")
            return res
        for n in names:
            self.add_column(n)
        self.apply_constraints()
        self.global_rules()
        for tgt, expr in self.rules.items():
            self._finish_rule(tgt, expr)
        for r in self.relationships:
            if r.get("source") == "assumed":
                r["source"] = "user"   # template correlations are listed under 'assumptions'; the schema field is kept simple
        self._finalize_columns()
        for name in sorted(self.placeholders):
            if self.cols[name]["type"] == "categorical" and self.cols[name].get("categories") == ["A", "B", "C"]:
                res.warnings.append(f"Column '{name}': allowed categories were not stated; placeholder categories A/B/C were "
                                    f"used. State them, e.g. '{name} can be Basic, Silver or Gold'.")
        self._report_unparsed()
        try:
            cols = []
            for s in self.cols.values():
                s = {k: v for k, v in s.items() if not k.startswith("_")}
                cols.append(ColumnSpec.from_dict(s))
            gen = {"n_rows": res.n_rows} if res.n_rows else {}
            res.schema = Schema(dataset_name=f"{self.domain or 'custom'}_dataset", columns=cols,
                                relationships=[r for r in self.relationships if self._corr_ok(r)],
                                constraints=self.constraints, generation=gen,
                                description="Schema interpreted from a plain-English description. Review before use.")
            res.schema.validate()
        except SchemaError as exc:
            res.warnings.append("The interpreted schema is not valid, so no schema was produced: " + str(exc).replace("\n", " "))
            res.schema = None
        if res.schema is not None and res.n_rows is None:
            res.warnings.append("No record count was found in the description; pass --rows / n_rows when generating.")
        return res

    def _corr_ok(self, r: dict) -> bool:
        for c in r["columns"]:
            s = self.cols.get(c)
            if s is None or s["type"] not in ("integer", "float", "date", "datetime") or c in self.rules \
                    or s.get("unique") or s.get("categories"):
                self.res.warnings.append(f"Correlation {r['columns']} dropped: both columns must be continuous, non-derived, non-unique numeric columns.")
                return False
        return True

    def _finish_rule(self, tgt: str, expr: str) -> None:
        spec = self.cols[tgt]
        # a derived column is defined only by its expression: drop generation properties inherited from the name hint
        for k in ("distribution", "mean", "std", "probability", "probabilities", "quantiles", "unique", "text_kind"):
            spec.pop(k, None)
        if spec["type"] not in ("integer", "float"):
            spec.pop("min", None)
            spec.pop("max", None)
        spec.pop("missing_rate", None)
        spec.pop("nullable", None)
        if spec["type"] == "categorical" and spec.get("categories") == ["Pass", "Fail"] and re.match(r"^\w+ (>=|>) ", expr):
            expr = f"where({expr}, 'Pass', 'Fail')"
        elif spec["type"] == "categorical" and re.match(r"^\w+ (>=|>|<=|<) ", expr):
            spec["type"] = "boolean"
            spec.pop("categories", None)
        spec["expression"] = expr
        refs = referenced_columns(expr)
        if any(self.cols.get(r, {}).get("missing_rate", 0) > 0 for r in refs):
            spec["nullable"] = True

    def _finalize_columns(self) -> None:
        for name, s in self.cols.items():
            if s["type"] == "float" and s.get("decimals") is None and not s.get("expression") and s.get("distribution") == "uniform":
                if isinstance(s.get("min"), int) and isinstance(s.get("max"), int):
                    s["decimals"] = 2
            if s["type"] == "integer" and s.get("expression"):
                s.pop("decimals", None)
            if s.get("missing_rate", 0) > 0:
                s["nullable"] = True
            if s["type"] == "identifier":
                s.pop("missing_rate", None)
                s.pop("nullable", None)

    def _report_unparsed(self) -> None:
        kw = re.compile(r"\b(?:must|should|between|at least|at most|correlat|if|when|depends?|unless|except|only|always|never|unique|greater|less)\b", re.I)
        for i, (a, b) in enumerate(self.sent_spans):
            s = self.text[a:b].strip()
            if i not in self.consumed and s and kw.search(s):
                self.res.unsupported.append(f"Not understood (no matching rule pattern): {s!r}")


def parse_description(text: str) -> DescriptionParseResult:
    """Interpret a plain-English dataset description as a reviewable schema.

    Never raises for unparseable text: check ``result.usable``, ``result.warnings`` and
    ``result.unsupported``. Save ``result.schema`` with :func:`dataseed.save_schema` to edit it.
    """
    if not isinstance(text, str):
        raise TypeError("parse_description expects a string.")
    return _Parser(text).build()
