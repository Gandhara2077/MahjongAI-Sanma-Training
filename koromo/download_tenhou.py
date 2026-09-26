#!/usr/bin/env python3
"""Download Tenhou Houou-table yonma (四鳳南) / sanma (三鳳南) hanchan game logs.

Multi-day ranges are downloaded as one global pipeline: all daily scc index
files are fetched concurrently first, then every referenced game log is pulled
through a single worker pool over persistent keep-alive HTTPS connections.
Tenhou serves game logs one HTTP request per game, so request count is fixed
by the game count; the rate limiter (seconds between request starts) remains
the politeness mechanism, while keep-alive removes the per-request TLS
handshake that previously dominated wall time.
"""

from __future__ import annotations

import argparse
import gzip
import http.client
import re
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable
from urllib.parse import urljoin, urlsplit
from urllib.request import Request, urlopen


LOG_ID_RE = re.compile(r"[?&]log=([0-9]{10}gm-[0-9a-z-]+)", re.IGNORECASE)
FILE_INDEX_RE = re.compile(
    r"\{file:'([^']*scc\d{8,10}\.html\.gz)',size:(\d+)\}"
)
FILE_INDEX_CACHE_SECONDS = 20 * 60
USER_AGENT = "koromo-tenhou-downloader/1.0"
RAW_BASE_URL = "https://tenhou.net/sc/raw/dat/"
RAW_HOST = "tenhou.net"
DOWNLOAD_VARIANTS = {
    "yonma": {"table_label": "四鳳南", "players": 4},
    "sanma": {"table_label": "三鳳南", "players": 3},
}
RETRYABLE_STATUS = {429, 503}
GAME_PROGRESS_EVERY = 500
TASK_CHUNK_SIZE = 4096


class RequestRateLimiter:
    """Serialize request starts while allowing downloads to overlap."""

    def __init__(self, delay: float) -> None:
        if delay < 0:
            raise ValueError("delay must be non-negative")
        self.delay = delay
        self._next_start = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            remaining = self._next_start - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            self._next_start = time.monotonic() + self.delay


class KeepAliveHTTPClient:
    """Thread-local persistent HTTPS connections to a single host.

    urllib.request performs a full TCP+TLS handshake per request; over range
    downloads of hundreds of thousands of game logs that handshake dominates
    wall time. Connections live per worker thread and are transparently
    re-established once when a kept-alive connection turns out to be stale.
    """

    def __init__(self, host: str, *, timeout: float = 60.0) -> None:
        self.host = host
        self.timeout = timeout
        self._local = threading.local()

    def _connection(self) -> http.client.HTTPSConnection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = http.client.HTTPSConnection(self.host, timeout=self.timeout)
            self._local.conn = conn
        return conn

    def fetch(self, path: str) -> tuple[int, dict[str, str], bytes]:
        """Single GET; returns (status, lowercase headers, body). No redirects."""
        for retry_on_stale in (True, False):
            conn = self._connection()
            try:
                conn.request(
                    "GET",
                    path,
                    headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"},
                )
                response = conn.getresponse()
                body = response.read()
                headers = {k.lower(): v for k, v in response.getheaders()}
                if response.will_close:
                    self._local.conn = None
                return response.status, headers, body
            except (http.client.HTTPException, OSError):
                # A pooled connection can be closed by the server between
                # requests; retry once on a fresh connection before failing.
                self._local.conn = None
                if not retry_on_stale:
                    raise
        raise AssertionError("unreachable")

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None


def parse_south_ids(html: str, table_label: str) -> list[str]:
    """Return unique log IDs from rows labelled table_label, preserving order."""
    ids: list[str] = []
    seen: set[str] = set()
    for line in html.splitlines():
        if table_label not in line:
            continue
        for match in LOG_ID_RE.finditer(line):
            log_id = match.group(1)
            if log_id not in seen:
                seen.add(log_id)
                ids.append(log_id)
    return ids


def parse_file_index(javascript: str) -> list[tuple[str, int]]:
    """Parse Tenhou's list.cgi JavaScript into (path, byte-size) pairs."""
    return [(path, int(size)) for path, size in FILE_INDEX_RE.findall(javascript)]


def is_complete_variant_mjlog(path: Path, players: int) -> bool:
    """Return whether path is a complete mjlog of the expected player count.

    The GO type bit field is parsed: the hanchan bit (0x08) must be set and
    the sanma bit (0x10) must match the variant's player count. Note this is
    stricter than the historical sanma downloader, which only checked the XML
    envelope; existing on-disk sanma logs are all 三鳳南 and stay valid.
    """
    try:
        data = path.read_bytes()
        if not data.startswith(b"<mjloggm ") or not data.rstrip().endswith(
            b"</mjloggm>"
        ):
            return False
        document = ET.fromstring(data)
        go = document.find("GO")
        game_type = int(go.attrib["type"]) if go is not None else -1
    except (OSError, ET.ParseError, KeyError, ValueError):
        return False

    is_hanchan = bool(game_type & 0x08)
    is_sanma = bool(game_type & 0x10)
    return is_hanchan and is_sanma == (players == 3)


def source_entries_for_date(
    entries: list[tuple[str, int]], date: str
) -> list[tuple[str, int]]:
    """Select daily or hourly scc files belonging to YYYYMMDD."""
    pattern = re.compile(rf"(?:^|/)scc{re.escape(date)}(?:\d{{2}})?\.html\.gz$")
    return [entry for entry in entries if pattern.search(entry[0])]


def latest_complete_date(entries: list[tuple[str, int]]) -> str:
    """Return the newest date represented by a complete daily scc file."""
    dates = [
        match.group(1)
        for path, _ in entries
        if (match := re.search(r"(?:^|/)scc(\d{8})\.html\.gz$", path))
    ]
    if not dates:
        raise ValueError("the official index contains no complete daily scc files")
    return max(dates)


def read_south_ids(paths: list[Path], table_label: str) -> list[str]:
    """Read compressed scc indexes and return unique South-game log IDs."""
    chunks: list[str] = []
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as file:
            chunks.append(file.read())
    return parse_south_ids("\n".join(chunks), table_label)


def mjlog_path(root: Path, log_id: str) -> Path:
    """Return the date-partitioned output path for a Tenhou log ID."""
    return (
        root
        / "mjlog"
        / log_id[0:4]
        / log_id[4:6]
        / log_id[6:8]
        / f"{log_id}.mjlog"
    )


def write_atomic(destination: Path, data: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)
    partial.write_bytes(data)
    partial.replace(destination)


def fetch_body(
    client: KeepAliveHTTPClient | None,
    url: str,
    *,
    rate_limiter: RequestRateLimiter | None,
    timeout: float,
    max_redirects: int = 4,
) -> bytes:
    """GET one URL, following redirects; raises OSError on HTTP errors."""
    for _ in range(max_redirects + 1):
        if rate_limiter is not None:
            rate_limiter.wait()
        parsed = urlsplit(url)
        use_client = (
            client is not None
            and parsed.scheme == "https"
            and parsed.netloc == client.host
        )
        if use_client:
            path = parsed.path + (f"?{parsed.query}" if parsed.query else "")
            status, headers, body = client.fetch(path)
        else:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=timeout) as response:
                status, body = response.status, response.read()
                headers = {k.lower(): v for k, v in response.headers.items()}
        if status in (301, 302, 307, 308):
            location = headers.get("location")
            if not location:
                raise OSError(f"HTTP {status} without Location for {url}")
            url = urljoin(url, location)
            continue
        if status >= 400:
            raise OSError(f"HTTP {status} for {url}")
        return body
    raise OSError(f"too many redirects for {url}")


def download_url(
    url: str,
    destination: Path,
    *,
    expected_size: int | None = None,
    validator: Callable[[Path], bool] | None = None,
    max_age: float | None = None,
    attempts: int = 4,
    timeout: float = 60.0,
    rate_limiter: RequestRateLimiter | None = None,
    client: KeepAliveHTTPClient | None = None,
) -> bool:
    """Atomically download a URL; return False when a valid file exists."""
    if destination.exists():
        if validator is not None and validator(destination):
            return False
        if (
            validator is None
            and expected_size is not None
            and destination.stat().st_size == expected_size
        ):
            return False
        if (
            validator is None
            and expected_size is None
            and max_age is not None
            and time.time() - destination.stat().st_mtime < max_age
        ):
            return False

    last_error: OSError | None = None
    for attempt in range(1, attempts + 1):
        try:
            body = fetch_body(
                client,
                url,
                rate_limiter=rate_limiter,
                timeout=timeout,
            )
            if expected_size is not None and len(body) != expected_size:
                raise OSError(
                    f"expected {expected_size} bytes from {url}, got {len(body)}"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_suffix(destination.suffix + ".part")
            partial.unlink(missing_ok=True)
            partial.write_bytes(body)
            if validator is not None and not validator(partial):
                raise OSError(f"downloaded content failed validation: {url}")
            partial.replace(destination)
            return True
        except OSError as error:
            last_error = error
            destination.with_suffix(destination.suffix + ".part").unlink(
                missing_ok=True
            )
            if attempt < attempts:
                time.sleep(min(2 ** (attempt - 1), 8))

    assert last_error is not None
    raise last_error


def ensure_scc_files(
    date: str,
    root: Path,
    entries: list[tuple[str, int]],
    *,
    client: KeepAliveHTTPClient,
    rate_limiter: RequestRateLimiter,
) -> list[Path]:
    """Download (or reuse) the scc source files covering YYYYMMDD."""
    selected = source_entries_for_date(entries, date)
    if not selected:
        raise ValueError(f"no official scc source file found for {date}")

    def fetch_one(item: tuple[str, int]) -> Path:
        relative_path, expected_size = item
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe source path in official index: {relative_path}")
        destination = root / "source" / relative
        download_url(
            RAW_BASE_URL + relative_path,
            destination,
            expected_size=expected_size,
            rate_limiter=rate_limiter,
            client=client,
        )
        return destination

    with ThreadPoolExecutor(max_workers=8) as executor:
        return list(executor.map(fetch_one, selected))


def write_date_manifest(root: Path, date: str, log_ids: list[str]) -> None:
    manifest = root / "manifests" / f"{date}.txt"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest_partial = manifest.with_suffix(".txt.part")
    manifest_partial.write_text(
        "".join(f"{log_id}\n" for log_id in log_ids), encoding="utf-8"
    )
    manifest_partial.replace(manifest)


def download_all(
    root: Path,
    tasks: list[tuple[str, str]],
    *,
    players: int,
    log_base_url: str = "https://tenhou.net/0/log/?",
    delay: float = 0.0,
    workers: int = 4,
    progress: Callable[[str], None] | None = None,
    rate_limiter: RequestRateLimiter | None = None,
    client: KeepAliveHTTPClient | None = None,
) -> dict[str, dict[str, int]]:
    """Download every (date, log_id) task through one shared worker pool."""
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive int")
    if rate_limiter is None:
        rate_limiter = RequestRateLimiter(delay)

    counters = {date: {"found": 0, "downloaded": 0, "skipped": 0} for date, _ in tasks}
    for date, _ in tasks:
        counters[date]["found"] += 1
    done = 0

    def download_log(item: tuple[str, str]) -> tuple[str, str, bool]:
        date, log_id = item
        was_downloaded = download_url(
            log_base_url + log_id,
            mjlog_path(root, log_id),
            validator=lambda path: is_complete_variant_mjlog(path, players),
            rate_limiter=rate_limiter,
            client=client,
        )
        return date, log_id, was_downloaded

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, len(tasks), TASK_CHUNK_SIZE):
            chunk = tasks[start : start + TASK_CHUNK_SIZE]
            for date, log_id, was_downloaded in executor.map(download_log, chunk):
                counters[date]["downloaded" if was_downloaded else "skipped"] += 1
                done += 1
                if progress is not None and done % GAME_PROGRESS_EVERY == 0:
                    progress(f"[{done}/{len(tasks)}] {log_id}")

    return counters


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--workers must be a positive integer"
        ) from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("--workers must be a positive integer")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download Tenhou Houou-table yonma/sanma hanchan mjlogs."
    )
    parser.add_argument(
        "--variant",
        choices=sorted(DOWNLOAD_VARIANTS),
        required=True,
        help="game variant to download (yonma = 四鳳南, sanma = 三鳳南)",
    )
    parser.add_argument(
        "--date",
        metavar="YYYYMMDD",
        help="date to download; defaults to the latest complete day",
    )
    parser.add_argument(
        "--start-date",
        metavar="YYYYMMDD",
        help="first date in an inclusive download range",
    )
    parser.add_argument(
        "--end-date",
        metavar="YYYYMMDD",
        help="last date in an inclusive download range",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="output root (default: directory containing this script)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="seconds between request starts (default: 0)",
    )
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=4,
        help="maximum concurrent log downloads (default: 4)",
    )
    parser.add_argument("--quiet", action="store_true", help="hide per-log progress")
    args = parser.parse_args(argv)

    if args.delay < 0:
        parser.error("--delay must be non-negative")
    for value, option in (
        (args.date, "--date"),
        (args.start_date, "--start-date"),
        (args.end_date, "--end-date"),
    ):
        if value is None:
            continue
        try:
            datetime.strptime(value, "%Y%m%d")
        except ValueError:
            parser.error(f"{option} must be a valid date in YYYYMMDD format")

    if args.date is not None and (
        args.start_date is not None or args.end_date is not None
    ):
        parser.error("--date cannot be combined with --start-date or --end-date")
    if (args.start_date is None) != (args.end_date is None):
        parser.error("--start-date and --end-date must be provided together")
    if (
        args.start_date is not None
        and datetime.strptime(args.end_date, "%Y%m%d")
        < datetime.strptime(args.start_date, "%Y%m%d")
    ):
        parser.error("--end-date must not be earlier than --start-date")

    root = args.output.resolve()
    variant = DOWNLOAD_VARIANTS[args.variant]
    rate_limiter = RequestRateLimiter(args.delay)
    client = KeepAliveHTTPClient(RAW_HOST)
    old_index_path = root / "source" / "file-index-old.js"
    download_url(
        "https://tenhou.net/sc/raw/list.cgi?old",
        old_index_path,
        max_age=FILE_INDEX_CACHE_SECONDS,
        rate_limiter=rate_limiter,
        client=client,
    )
    entries = parse_file_index(old_index_path.read_text(encoding="utf-8"))

    if args.start_date is not None and args.end_date is not None:
        start = datetime.strptime(args.start_date, "%Y%m%d")
        end = datetime.strptime(args.end_date, "%Y%m%d")
        selected_dates = [
            (start + timedelta(days=offset)).strftime("%Y%m%d")
            for offset in range((end - start).days + 1)
        ]
    else:
        selected_dates = [args.date or latest_complete_date(entries)]

    missing_dates = [
        date for date in selected_dates if not source_entries_for_date(entries, date)
    ]
    if missing_dates:
        current_index_path = root / "source" / "file-index-current.js"
        download_url(
            "https://tenhou.net/sc/raw/list.cgi",
            current_index_path,
            max_age=FILE_INDEX_CACHE_SECONDS,
            rate_limiter=rate_limiter,
            client=client,
        )
        current_entries = parse_file_index(
            current_index_path.read_text(encoding="utf-8")
        )
        entries.extend(current_entries)
        still_missing = [
            date for date in selected_dates if not source_entries_for_date(entries, date)
        ]
        if still_missing:
            raise ValueError(
                f"no official scc source files for: {', '.join(still_missing[:5])}"
                + (f" (+{len(still_missing) - 5} more)" if len(still_missing) > 5 else "")
            )

    # Phase A: concurrent scc index downloads for every requested date.
    source_paths_by_date: dict[str, list[Path]] = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            date: executor.submit(
                ensure_scc_files,
                date,
                root,
                entries,
                client=client,
                rate_limiter=rate_limiter,
            )
            for date in selected_dates
        }
        for date, future in futures.items():
            source_paths_by_date[date] = future.result()

    # Phase B: parse log IDs and refresh per-date manifests.
    tasks: list[tuple[str, str]] = []
    for date in selected_dates:
        log_ids = read_south_ids(
            source_paths_by_date[date], variant["table_label"]
        )
        write_date_manifest(root, date, log_ids)
        tasks.extend((date, log_id) for log_id in log_ids)

    progress = None if args.quiet else print
    if len(selected_dates) == 1:
        print(
            f"Downloading {variant['table_label']} logs for {selected_dates[0]} "
            f"into {root}: {len(tasks)} games"
        )
    else:
        print(
            f"Downloading {variant['table_label']} logs for "
            f"{selected_dates[0]}..{selected_dates[-1]} into {root}: "
            f"{len(selected_dates)} days, {len(tasks)} games"
        )

    # Phase C: one global keep-alive worker pool over all referenced games.
    counters = download_all(
        root,
        tasks,
        players=variant["players"],
        delay=args.delay,
        workers=args.workers,
        progress=progress,
        rate_limiter=rate_limiter,
        client=client,
    )

    totals = {"found": 0, "downloaded": 0, "skipped": 0}
    for date in selected_dates:
        summary = counters[date]
        if len(selected_dates) == 1 or summary["downloaded"]:
            print(
                f"Finished {date}: "
                f"{summary['found']} found, {summary['downloaded']} downloaded, "
                f"{summary['skipped']} already present"
            )
        for key in totals:
            totals[key] += summary[key]
    if len(selected_dates) > 1:
        print(
            f"Finished range {selected_dates[0]}..{selected_dates[-1]}: "
            f"{totals['found']} found, {totals['downloaded']} downloaded, "
            f"{totals['skipped']} already present"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
