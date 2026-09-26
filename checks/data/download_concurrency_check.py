"""Check bounded log-download concurrency without contacting Tenhou.

Exercises the current koromo.download_tenhou API (ensure_scc_files,
read_south_ids, write_date_manifest and download_all sharing one
RequestRateLimiter) against fake transports: the keep-alive client used for
the official scc index files and the urlopen() fallback used for game logs.
"""

from __future__ import annotations

import gzip
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from koromo import download_tenhou


LOG_IDS = [
    "2026010100gm-00b9-0000-000000000001",
    "2026010100gm-00b9-0000-000000000002",
    "2026010100gm-00b9-0000-000000000003",
]
DATE = "20260101"
SANMA = download_tenhou.DOWNLOAD_VARIANTS["sanma"]
# GO type 24 = 0x18: hanchan bit (0x08) + sanma bit (0x10).
SANMA_LOG_BODY = b'<mjloggm ><GO type="24" /></mjloggm>'
YONMA_LOG_BODY = b'<mjloggm ><GO type="9" /></mjloggm>'
LOG_BASE_URL = "https://logs/?"


class ActivityMeter:
    """Track how many fake log downloads are in flight at the same time."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active = 0
        self.maximum = 0

    def enter(self) -> None:
        with self.lock:
            self.active += 1
            self.maximum = max(self.maximum, self.active)

    def exit(self) -> None:
        with self.lock:
            self.active -= 1


class FakeResponse:
    """Minimal stand-in for the urlopen() response consumed by fetch_body."""

    def __init__(self, body: bytes, *, meter: ActivityMeter | None = None) -> None:
        self.status = 200
        self.headers = {"Content-Type": "text/plain"}
        self._body = body
        self._meter = meter

    def __enter__(self) -> "FakeResponse":
        if self._meter is not None:
            self._meter.enter()
        return self

    def __exit__(self, *_: object) -> None:
        if self._meter is not None:
            self._meter.exit()

    def read(self) -> bytes:
        if self._meter is not None:
            # Keep the request in flight long enough for workers to overlap.
            time.sleep(0.12)
        return self._body


class FakeKeepAliveClient:
    """Stand-in for KeepAliveHTTPClient serving the official scc index."""

    host = download_tenhou.RAW_HOST

    def __init__(self, body: bytes) -> None:
        self.body = body
        self.paths: list[str] = []

    def fetch(self, path: str) -> tuple[int, dict[str, str], bytes]:
        self.paths.append(path)
        return 200, {"content-length": str(len(self.body))}, self.body

    def close(self) -> None:
        pass


def check_parsing_and_validation(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    label = SANMA["table_label"]
    html = label + " " + " ".join(f"?log={log_id}" for log_id in LOG_IDS)
    assert download_tenhou.parse_south_ids(html, label) == LOG_IDS
    # Rows of other tables and repeated IDs are ignored.
    noisy = html + "\n四鳳南 ?log=2026010100gm-00a9-0000-000000000009\n" + html
    assert download_tenhou.parse_south_ids(noisy, label) == LOG_IDS

    sanma_log = root / "sanma.mjlog"
    yonma_log = root / "yonma.mjlog"
    sanma_log.write_bytes(SANMA_LOG_BODY)
    yonma_log.write_bytes(YONMA_LOG_BODY)
    assert download_tenhou.is_complete_variant_mjlog(sanma_log, 3)
    assert not download_tenhou.is_complete_variant_mjlog(sanma_log, 4)
    assert download_tenhou.is_complete_variant_mjlog(yonma_log, 4)
    assert not download_tenhou.is_complete_variant_mjlog(yonma_log, 3)
    (root / "truncated.mjlog").write_bytes(SANMA_LOG_BODY[:-4])
    assert not download_tenhou.is_complete_variant_mjlog(root / "truncated.mjlog", 3)


def check_source_files(root: Path, source_body: bytes) -> list[Path]:
    entries = [(f"scc{DATE}.html.gz", len(source_body))]
    client = FakeKeepAliveClient(source_body)
    limiter = download_tenhou.RequestRateLimiter(0.0)
    sources = download_tenhou.ensure_scc_files(
        DATE, root, entries, client=client, rate_limiter=limiter
    )
    assert sources == [root / "source" / f"scc{DATE}.html.gz"], sources
    assert sources[0].stat().st_size == len(source_body)
    assert client.paths == [f"/sc/raw/dat/scc{DATE}.html.gz"], client.paths
    # A complete source file is reused instead of being downloaded again.
    again = download_tenhou.ensure_scc_files(
        DATE, root, entries, client=client, rate_limiter=limiter
    )
    assert again == sources and len(client.paths) == 1
    assert not list(root.rglob("*.part"))

    try:
        download_tenhou.ensure_scc_files(
            DATE, root, [(f"../scc{DATE}.html.gz", 1)],
            client=client, rate_limiter=limiter,
        )
    except ValueError as error:
        assert "unsafe source path" in str(error), error
    else:
        raise AssertionError("path traversal in the official index was accepted")

    assert download_tenhou.read_south_ids(sources, SANMA["table_label"]) == LOG_IDS
    download_tenhou.write_date_manifest(root, DATE, LOG_IDS)
    manifest = root / "manifests" / f"{DATE}.txt"
    assert manifest.read_text(encoding="utf-8").split() == LOG_IDS
    assert not list(root.rglob("*.part"))
    return sources


def check_concurrent_downloads(root: Path) -> None:
    meter = ActivityMeter()
    request_times: list[float] = []

    def fake_urlopen(request: object, timeout: float = 0.0) -> FakeResponse:
        del timeout
        url = request.full_url  # type: ignore[attr-defined]
        assert url.startswith(LOG_BASE_URL), url
        assert request.get_header("User-agent") == download_tenhou.USER_AGENT  # type: ignore[attr-defined]
        request_times.append(time.monotonic())
        return FakeResponse(SANMA_LOG_BODY, meter=meter)

    tasks = [(DATE, log_id) for log_id in LOG_IDS]
    download_tenhou.urlopen = fake_urlopen
    counters = download_tenhou.download_all(
        root, tasks, players=SANMA["players"], log_base_url=LOG_BASE_URL,
        delay=0.06, workers=3, client=None,
    )
    assert counters == {DATE: {"found": 3, "downloaded": 3, "skipped": 0}}, counters
    assert meter.maximum > 1, "workers did not overlap downloads"
    assert meter.active == 0
    assert len(request_times) == 3, request_times
    timestamps = sorted(request_times)
    assert all(
        later - earlier >= 0.03
        for earlier, later in zip(timestamps, timestamps[1:])
    ), f"global request spacing was too short: {timestamps}"
    for log_id in LOG_IDS:
        destination = download_tenhou.mjlog_path(root, log_id)
        assert destination.read_bytes() == SANMA_LOG_BODY
        assert download_tenhou.is_complete_variant_mjlog(destination, 3)
    assert not list(root.rglob("*.part"))

    # Valid on-disk logs are skipped without any request.
    request_times.clear()
    counters = download_tenhou.download_all(
        root, tasks, players=SANMA["players"], log_base_url=LOG_BASE_URL,
        delay=0.06, workers=3, client=None,
    )
    assert counters == {DATE: {"found": 3, "downloaded": 0, "skipped": 3}}, counters
    assert request_times == []


def check_failure_path(root: Path) -> None:
    retry_sleeps: list[float] = []

    def failing_urlopen(request: object, timeout: float = 0.0) -> FakeResponse:
        del request, timeout
        raise OSError("synthetic log failure")

    original_sleep = download_tenhou.time.sleep
    download_tenhou.urlopen = failing_urlopen
    download_tenhou.time.sleep = lambda seconds: retry_sleeps.append(seconds)
    try:
        try:
            download_tenhou.download_all(
                root, [(DATE, LOG_IDS[0])], players=SANMA["players"],
                log_base_url=LOG_BASE_URL, delay=0.0, workers=1, client=None,
            )
        except OSError as error:
            assert str(error) == "synthetic log failure", error
        else:
            raise AssertionError("download failure was swallowed")
    finally:
        download_tenhou.time.sleep = original_sleep
    assert retry_sleeps == [1, 2, 4], retry_sleeps
    assert not download_tenhou.mjlog_path(root, LOG_IDS[0]).exists()
    assert not list(root.rglob("*.part"))

    for workers in (0, -1, True):
        try:
            download_tenhou.download_all(
                root, [], players=SANMA["players"], workers=workers,  # type: ignore[arg-type]
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"workers={workers!r} was accepted")


def main() -> None:
    source_text = (
        SANMA["table_label"]
        + " "
        + " ".join(f"?log={log_id}" for log_id in LOG_IDS)
    )
    source_body = gzip.compress(source_text.encode("utf-8"))
    original_urlopen = download_tenhou.urlopen
    try:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            check_parsing_and_validation(root / "parse")
            check_source_files(root / "source_files", source_body)
            check_concurrent_downloads(root / "downloads")
        with tempfile.TemporaryDirectory() as temporary:
            check_failure_path(Path(temporary))
    finally:
        download_tenhou.urlopen = original_urlopen
    print("DOWNLOAD_CONCURRENCY_OK")


if __name__ == "__main__":
    main()
