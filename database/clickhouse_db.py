import logging
import uuid

from config import Config
logger = logging.getLogger(__name__)

try:
    import clickhouse_connect
except ImportError:
    clickhouse_connect = None

from config import Config


class ClickHouseManager:
    def __init__(self):
        self.client = None
        self.last_error = None
        self._connect()

    @property
    def configured(self) -> bool:
        return Config.clickhouse_configured()

    def _connect(self):
        if not self.configured:
            print("[ClickHouse] Not configured; database features disabled.")
            return

        if clickhouse_connect is None:
            self.last_error = "clickhouse-connect package is not installed"
            print(f"[ClickHouse] {self.last_error}")
            return

        try:
            self.client = clickhouse_connect.get_client(
                host=Config.CLICKHOUSE_HOST,
                port=Config.CLICKHOUSE_PORT,
                username=Config.CLICKHOUSE_USER,
                password=Config.CLICKHOUSE_PASSWORD,
                database=Config.CLICKHOUSE_DATABASE,
                secure=True,
            )

            self.client.command("SELECT 1")
            self._initialize_tables()
            print("[ClickHouse] Successfully connected.")

        except Exception as exc:
            self.last_error = str(exc)
            print(f"[ClickHouse Error] Connection failed: {self.last_error}")
            self.client = None

    def _initialize_tables(self):
        # 1. Main scan results
        self.client.command("""
            CREATE TABLE IF NOT EXISTS scans (
                scan_id UUID,
                input_type String,
                source_path String,
                media_type String,
                verdict String,
                confidence_score Float32,
                scanned_at DateTime DEFAULT now()
            )
            ENGINE = MergeTree()
            ORDER BY (scanned_at, scan_id)
        """)

        # 2. Per-agent telemetry
        self.client.command("""
            CREATE TABLE IF NOT EXISTS detection_telemetry (
                session_id String,
                timestamp DateTime DEFAULT now(),
                agent_name String,
                anomaly_score Float32,
                status String,
                details String
            )
            ENGINE = MergeTree()
            ORDER BY (timestamp, session_id)
        """)

        # 3. Takedown requests
        self.client.command("""
            CREATE TABLE IF NOT EXISTS takedown_requests (
                request_id UUID,
                scan_id Nullable(UUID),
                platform String,
                target_url String,
                status String,
                created_at DateTime DEFAULT now()
            )
            ENGINE = MergeTree()
            ORDER BY (created_at, request_id)
        """)

        # 4. System logs
        self.client.command("""
            CREATE TABLE IF NOT EXISTS system_logs (
                log_id UUID,
                event_type String,
                execution_time_ms UInt32,
                status String,
                timestamp DateTime DEFAULT now()
            )
            ENGINE = MergeTree()
            ORDER BY (timestamp, log_id)
        """)

        print("[ClickHouse] Tables initialized.")

    # ---------------------------------------------------------
    # SCANS
    # ---------------------------------------------------------

    def save_scan(
        self,
        scan_id,
        input_type,
        source_path,
        media_type,
        verdict,
        confidence_score,
    ) -> bool:
        if not self.client:
            return False

        try:
            self.client.insert(
                "scans",
                [[
                    str(scan_id),
                    input_type,
                    source_path,
                    media_type,
                    verdict,
                    float(confidence_score),
                ]],
                column_names=[
                    "scan_id",
                    "input_type",
                    "source_path",
                    "media_type",
                    "verdict",
                    "confidence_score",
                ],
            )

            return True

        except Exception as exc:
            print(f"[ClickHouse Scan Error] {exc}")
            return False

    def find_previous_scan(self, source_path: str):
        if not self.client:
            return None

        try:
            result = self.client.query(
                """
                SELECT
                    scan_id,
                    input_type,
                    source_path,
                    media_type,
                    verdict,
                    confidence_score,
                    scanned_at
                FROM scans
                WHERE source_path = {source_path:String}
                ORDER BY scanned_at DESC
                LIMIT 1
                """,
                parameters={"source_path": source_path},
            )

            if result.result_rows:
                row = result.result_rows[0]

                return {
                    "scan_id": str(row[0]),
                    "input_type": row[1],
                    "source_path": row[2],
                    "media_type": row[3],
                    "verdict": row[4],
                    "confidence_score": float(row[5]),
                    "scanned_at": row[6],
                }

            return None

        except Exception as exc:
            print(f"[ClickHouse Lookup Error] {exc}")
            return None

    # ---------------------------------------------------------
    # AGENT TELEMETRY
    # ---------------------------------------------------------

    def log_agent_execution(
        self,
        session_id: str,
        agent_name: str,
        anomaly_score: float,
        status: str,
        details: str,
    ) -> bool:
        if not self.client:
            return False

        try:
            self.client.insert(
                "detection_telemetry",
                [[
                    session_id,
                    agent_name,
                    float(anomaly_score),
                    status,
                    str(details),
                ]],
                column_names=[
                    "session_id",
                    "agent_name",
                    "anomaly_score",
                    "status",
                    "details",
                ],
            )

            return True

        except Exception as exc:
            print(f"[ClickHouse Agent Log Error] {exc}")
            return False

    # ---------------------------------------------------------
    # TAKEDOWN REQUESTS
    # ---------------------------------------------------------

    def save_takedown_request(
        self,
        request_id,
        scan_id,
        platform: str,
        target_url: str,
        status: str = "PENDING",
    ) -> bool:
        if not self.client:
            return False

        try:
            self.client.insert(
                "takedown_requests",
                [[
                    str(request_id),
                    str(scan_id) if scan_id else None,
                    platform,
                    target_url,
                    status,
                ]],
                column_names=[
                    "request_id",
                    "scan_id",
                    "platform",
                    "target_url",
                    "status",
                ],
            )

            return True

        except Exception as exc:
            print(f"[ClickHouse Takedown Error] {exc}")
            return False

    def log_takedown_request(
        self,
        request_id,
        target_url,
        similarity_score,
        notice_type,
        action_status="PENDING",
    ):
        """Compatibility method for the existing takedown agent."""
        try:
            self.client.insert(
                "takedown_requests",
                [[
                    str(request_id),
                    None,
                    notice_type or "UNKNOWN",
                    target_url,
                    action_status or "PENDING",
                ]],
                column_names=[
                    "request_id",
                    "scan_id",
                    "platform",
                    "target_url",
                    "status",
                ],
            )

            print("[ClickHouse] Takedown request logged successfully.")
            return True

        except Exception as exc:
            print(f"[ClickHouse Takedown Error] {exc}")
            return False

    # ---------------------------------------------------------
    # SYSTEM LOGS
    # ---------------------------------------------------------

    def log_system_event(
        self,
        event_type: str,
        execution_time_ms: int = 0,
        status: str = "SUCCESS",
    ) -> bool:
        if not self.client:
            return False

        try:
            self.client.insert(
                "system_logs",
                [[
                    str(uuid.uuid4()),
                    event_type,
                    int(execution_time_ms),
                    status,
                ]],
                column_names=[
                    "log_id",
                    "event_type",
                    "execution_time_ms",
                    "status",
                ],
            )

            return True

        except Exception as exc:
            print(f"[ClickHouse System Log Error] {exc}")
            return False


# Global database manager
db_manager = ClickHouseManager()