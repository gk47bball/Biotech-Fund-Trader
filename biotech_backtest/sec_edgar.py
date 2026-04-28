"""SEC EDGAR client: list 13F-HR filings for a fund and parse the info table."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

import requests

from .config import FILINGS_CACHE, APIConfig

SEC_BASE = "https://www.sec.gov"
SEC_DATA = "https://data.sec.gov"


@dataclass
class Holding:
    cusip: str
    issuer: str
    title: str
    value_usd: float       # 13F values are reported in dollars (post-2022) or thousands (pre-2022)
    shares: float
    put_call: str | None = None  # 'Put', 'Call', or None for shares


@dataclass
class Filing:
    fund: str
    cik: str
    accession: str         # accession number with dashes
    report_date: str       # period of report YYYY-MM-DD (quarter end)
    filing_date: str       # date filed YYYY-MM-DD
    holdings: list[Holding]


def _session(api: APIConfig) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": api.sec_user_agent,
        "Accept-Encoding": "gzip, deflate",
        "Host": None,
    })
    return s


def _get(session: requests.Session, url: str, retries: int = 4) -> requests.Response:
    """SEC enforces ~10 req/s; we throttle and retry."""
    delay = 0.15
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=30)
            if r.status_code == 200:
                time.sleep(delay)
                return r
            if r.status_code in (403, 429):
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
        except requests.RequestException as e:
            last_exc = e
            time.sleep(2 ** attempt)
    if last_exc:
        raise last_exc
    raise RuntimeError(f"GET failed after {retries} retries: {url}")


def list_13f_filings(cik: str, api: APIConfig) -> list[dict]:
    """Return list of {accession, filing_date, report_date, primary_doc} for all 13F-HR filings."""
    cik_padded = cik.lstrip("0").zfill(10)
    session = _session(api)
    out: list[dict] = []

    # Initial submissions JSON contains the most recent ~1000 filings.
    url = f"{SEC_DATA}/submissions/CIK{cik_padded}.json"
    r = _get(session, url)
    payload = r.json()
    out.extend(_parse_submissions(payload.get("filings", {}).get("recent", {})))

    # Older filings live in linked files.
    for file_entry in payload.get("filings", {}).get("files", []):
        ru = f"{SEC_DATA}/submissions/{file_entry['name']}"
        rr = _get(session, ru)
        out.extend(_parse_submissions(rr.json()))

    return [f for f in out if f["form"] in ("13F-HR", "13F-HR/A")]


def _parse_submissions(recent: dict) -> list[dict]:
    rows: list[dict] = []
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", [])
    primary_docs = recent.get("primaryDocument", [])
    for i, form in enumerate(forms):
        rows.append({
            "form": form,
            "accession": accessions[i],
            "filing_date": filing_dates[i],
            "report_date": report_dates[i],
            "primary_doc": primary_docs[i] if i < len(primary_docs) else "",
        })
    return rows


def fetch_filing(fund: str, cik: str, accession: str, filing_date: str, report_date: str,
                 api: APIConfig) -> Filing:
    """Download and parse the info table for one 13F filing."""
    cache_path = FILINGS_CACHE / f"{cik}_{accession.replace('-', '')}.json"
    if cache_path.exists():
        data = json.loads(cache_path.read_text())
        return Filing(
            fund=fund,
            cik=cik,
            accession=accession,
            report_date=data["report_date"],
            filing_date=data["filing_date"],
            holdings=[Holding(**h) for h in data["holdings"]],
        )

    session = _session(api)
    cik_int = str(int(cik))
    acc_no_dashes = accession.replace("-", "")
    base = f"{SEC_BASE}/Archives/edgar/data/{cik_int}/{acc_no_dashes}"

    # Find the info-table XML by fetching the filing index JSON.
    idx = _get(session, f"{base}/").text
    info_table_name = _find_info_table(idx)
    if info_table_name is None:
        # Empty filing or unusual format — treat as no holdings.
        filing = Filing(fund=fund, cik=cik, accession=accession,
                        report_date=report_date, filing_date=filing_date, holdings=[])
        _write_cache(cache_path, filing)
        return filing

    xml_text = _get(session, f"{base}/{info_table_name}").text
    holdings = _parse_info_table(xml_text)
    filing = Filing(fund=fund, cik=cik, accession=accession,
                    report_date=report_date, filing_date=filing_date, holdings=holdings)
    _write_cache(cache_path, filing)
    return filing


def _find_info_table(index_html: str) -> str | None:
    """The info table is an XML doc, usually named *infotable*.xml."""
    candidates = re.findall(r'href="[^"]*?/([^/"]+\.xml)"', index_html, flags=re.I)
    for name in candidates:
        if "infotable" in name.lower() or "info_table" in name.lower():
            return name
    # Fallback: any xml that's not the primary 13F submission summary.
    for name in candidates:
        if not name.lower().startswith("primary_doc"):
            return name
    return None


_NS_RE = re.compile(r"\{[^}]+\}")


def _strip_ns(tag: str) -> str:
    return _NS_RE.sub("", tag)


def _parse_info_table(xml_text: str) -> list[Holding]:
    root = ET.fromstring(xml_text)
    holdings: list[Holding] = []
    for info in root.iter():
        if _strip_ns(info.tag) != "infoTable":
            continue
        fields: dict[str, str] = {}
        for child in info.iter():
            t = _strip_ns(child.tag)
            if child.text and child.text.strip():
                fields[t] = child.text.strip()
        try:
            value = float(fields.get("value", "0").replace(",", ""))
            shares = float(fields.get("sshPrnamt", "0").replace(",", ""))
        except ValueError:
            continue
        holdings.append(Holding(
            cusip=fields.get("cusip", "").strip().upper(),
            issuer=fields.get("nameOfIssuer", ""),
            title=fields.get("titleOfClass", ""),
            value_usd=value,
            shares=shares,
            put_call=fields.get("putCall") or None,
        ))
    return holdings


def _write_cache(path: Path, filing: Filing) -> None:
    path.write_text(json.dumps({
        "fund": filing.fund,
        "cik": filing.cik,
        "accession": filing.accession,
        "report_date": filing.report_date,
        "filing_date": filing.filing_date,
        "holdings": [h.__dict__ for h in filing.holdings],
    }))


def fetch_fund_filings(fund: str, cik: str, start_date: str, end_date: str,
                       api: APIConfig) -> list[Filing]:
    """Fetch + parse all 13F-HR filings for a fund within [start_date, end_date] by report_date."""
    listings = list_13f_filings(cik, api)
    listings = [
        f for f in listings
        if start_date <= f["report_date"] <= end_date
    ]
    listings.sort(key=lambda f: f["report_date"])

    # If a quarter has both 13F-HR and a later 13F-HR/A, prefer the amendment.
    by_qtr: dict[str, dict] = {}
    for f in listings:
        existing = by_qtr.get(f["report_date"])
        if existing is None or f["filing_date"] > existing["filing_date"]:
            by_qtr[f["report_date"]] = f
    deduped = sorted(by_qtr.values(), key=lambda f: f["report_date"])

    filings: list[Filing] = []
    for f in deduped:
        filings.append(fetch_filing(
            fund=fund, cik=cik,
            accession=f["accession"], filing_date=f["filing_date"], report_date=f["report_date"],
            api=api,
        ))
    return filings


def fetch_all_funds(funds: dict[str, str], start_date: str, end_date: str,
                    api: APIConfig) -> dict[str, list[Filing]]:
    out: dict[str, list[Filing]] = {}
    for fund, cik in funds.items():
        out[fund] = fetch_fund_filings(fund, cik, start_date, end_date, api)
    return out
