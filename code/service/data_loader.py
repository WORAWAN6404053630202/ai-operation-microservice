# code/service/data_loader.py
"""
Data Loader Adapter
Loads data from Google Sheets and converts to document format

PRODUCTION (RAG behavior):
- Reduce page_content noise to improve embedding/retrieval precision.
- Put high-signal procedural fields in page_content (steps/docs/fees/channel/legal/duration/conditions/topic)
- Keep everything else in metadata (still queryable downstream if needed).
"""

import re
import math
import pandas as pd
from langchain_core.documents import Document
from typing import List, Optional, Dict
from urllib.parse import urlparse, parse_qs


class DataLoader:
    """
    Flexible Data Loader
    - Load Google Sheet via CSV export URL
    - Normalize multi-line headers
    - Convert NaN/None → None
    - Build Document + metadata compatible with pipeline

    PATCH (production):
    - Robust header alias mapping for Thai government sheets where column names change often.
    - Supports:
        * exact header
        * alias list
        * "contains keyword" fallback

    PATCH (RAG quality):
    - page_content: only include high-signal procedural/legal fields to reduce embedding drift.
    - other columns remain in metadata.
    """

    def __init__(self, config):
        self.config = config
        self.documents = []
        self.departments_found = set()

        # RAG content tuning knobs (safe defaults)
        self.page_content_max_chars = int(getattr(config, "PAGE_CONTENT_MAX_CHARS", 1800) or 1800)
        self.include_research_reference_in_content = bool(
            getattr(config, "INCLUDE_RESEARCH_REF_IN_CONTENT", False)
        )

    @staticmethod
    def clean_header(name: str) -> str:
        if not isinstance(name, str):
            return name
        name = name.replace("\n", " ").replace("\r", " ")
        name = re.sub(r"\s+", " ", name)
        return name.strip()

    @staticmethod
    def _build_csv_export_url(sheet_url: str) -> str:
        """
        Build Google Sheets CSV export URL robustly.
        - Accepts URLs like:
          /edit?gid=657...#gid=657...
          /edit#gid=657...
          /edit?usp=sharing&gid=657...
        - Extracts spreadsheet base and gid safely.
        """
        u = urlparse(sheet_url)

        # Base path: https://docs.google.com/spreadsheets/d/<SPREADSHEET_ID>
        base = f"{u.scheme}://{u.netloc}{u.path}"
        base = base.split("/edit")[0].rstrip("/")

        # gid can be in query (?gid=) or fragment (#gid=)
        q = parse_qs(u.query)
        gid = (q.get("gid", [None])[0]) or (parse_qs(u.fragment).get("gid", [None])[0])

        if not gid:
            raise ValueError(
                "Google Sheet URL missing gid. "
                "Please provide a link that includes gid=... (pointing to the content sheet tab)."
            )
        print("[DataLoader] Using gid:", gid)
        return f"{base}/export?format=csv&gid={gid}"

    @staticmethod
    def to_json_safe(v):
        if v is None:
            return None

        try:
            if isinstance(v, float) and math.isnan(v):
                return None
        except Exception:
            pass

        s = str(v).strip()
        return s if s and s.lower() != "nan" else None

    def load_from_google_sheet(self, sheet_url: str, source_name: str = None):
        print("\nLoading data from Google Sheet...")
        csv_url = self._build_csv_export_url(sheet_url)
        print("CSV export URL:", csv_url)
        df = pd.read_csv(csv_url)
        df.rename(columns={c: self.clean_header(c) for c in df.columns}, inplace=True)
        # NEW: schema validation (fail fast)
        self._validate_sheet_schema(df, sheet_url)

        print("Columns detected:")
        for c in df.columns:
            print(" •", repr(c))

        print(f"\nLoaded {len(df)} rows")
        docs_added = self._process_dataframe(df, source=source_name or csv_url)

        print(f"Added {docs_added} documents")
        return docs_added

    # -----------------------------
    # Header resolution (NEW)
    # -----------------------------
    def _resolve_col(
        self,
        df: pd.DataFrame,
        *,
        primary: str,
        aliases: Optional[List[str]] = None,
        contains_any: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Resolve dataframe column name robustly.
        Priority:
          1) exact match: primary
          2) exact match: any alias
          3) contains-match: any token in contains_any (case-insensitive)
        Returns actual column name in df or None.
        """
        cols = list(df.columns)
        if primary in cols:
            return primary

        for a in (aliases or []):
            if a in cols:
                return a

        if contains_any:
            low_cols = [(c, str(c).lower()) for c in cols]
            for token in contains_any:
                t = str(token).lower().strip()
                if not t:
                    continue
                for orig, low in low_cols:
                    if t in low:
                        return orig

        return None

    def _build_column_map(self, df: pd.DataFrame) -> Dict[str, Optional[str]]:
        """
        Build mapping from logical field -> actual df column name.
        This keeps ingestion stable even if sheet headers change slightly.
        """
        return {
            "department": self._resolve_col(
                df,
                primary="หน่วยงาน",
                aliases=[],
                contains_any=["หน่วยงาน"],
            ),
            "license_type": self._resolve_col(
                df,
                primary="ใบอนุญาต",
                aliases=[],
                contains_any=["ใบอนุญาต"],
            ),
            "operation_by_department": self._resolve_col(
                df,
                primary="การดำเนินการ ตามหน่วยงาน",
                aliases=["การดำเนินการตามหน่วยงาน", "การดำเนินการ  ตามหน่วยงาน"],
                contains_any=["การดำเนินการ", "ตามหน่วยงาน"],
            ),
            "operation_topic": self._resolve_col(
                df,
                primary="หัวข้อการดำเนินการ",
                aliases=["หัวข้อการดำเนินการย่อย", "หัวข้อการดำเนินการ (ย่อย)"],
                contains_any=["หัวข้อการดำเนินการ"],
            ),
            "registration_type": self._resolve_col(
                df,
                primary="ประเภทการจดทะเบียน",
                aliases=[],
                contains_any=["ประเภทการจดทะเบียน", "ประเภท", "จดทะเบียน"],
            ),
            "terms_and_conditions": self._resolve_col(
                df,
                primary="เงื่อนไขและหลักเกณฑ์",
                aliases=["เงื่อนไข", "หลักเกณฑ์"],
                contains_any=["เงื่อนไข", "หลักเกณฑ์"],
            ),
            "service_channel": self._resolve_col(
                df,
                primary="ช่องทางการ ให้บริการ",
                aliases=["ช่องทางการให้บริการ", "ช่องทางให้บริการ", "ช่องทาง"],
                contains_any=["ช่องทาง", "ให้บริการ"],
            ),
            "operation_steps": self._resolve_col(
                df,
                primary="ขั้นตอนการดำเนินการ",
                aliases=["ขั้นตอน"],
                contains_any=["ขั้นตอนการดำเนินการ", "ขั้นตอน"],
            ),
            "identification_documents": self._resolve_col(
                df,
                primary="เอกสาร ยืนยันตัวตน",
                aliases=["เอกสารยืนยันตัวตน", "เอกสาร ยืนยันตัวตน", "เอกสารยืนยัน"],
                contains_any=["เอกสาร", "ยืนยันตัวตน"],
            ),
            "operation_duration": self._resolve_col(
                df,
                primary="ระยะเวลา การดำเนินการ",
                aliases=["ระยะเวลา การดำเนินการ", "ระยะเวลา", "ระยะเวลาการดำเนินการ"],
                contains_any=["ระยะเวลา"],
            ),
            "fees": self._resolve_col(
                df,
                primary="ค่าธรรมเนียม",
                aliases=[],
                contains_any=["ค่าธรรมเนียม", "ค่าใช้จ่าย", "ค่าบริการ"],
            ),
            "legal_regulatory": self._resolve_col(
                df,
                primary="ข้อกำหนดทางกฎหมาย และข้อบังคับ",
                aliases=["ข้อกำหนดทางกฎหมายและข้อบังคับ", "ข้อกำหนดทางกฎหมาย", "ข้อบังคับ"],
                contains_any=["ข้อกำหนด", "กฎหมาย", "ข้อบังคับ"],
            ),
            "restaurant_ai_document": self._resolve_col(
                df,
                primary="เอกสาร AI ร้านอาหาร",
                aliases=["เอกสาร AI ร้านอาหาร", "เอกสาร AI", "เอกสาร (AI)"],
                contains_any=["เอกสาร ai", "เอกสาร (ai)"],
            ),
            "research_reference": self._resolve_col(
                df,
                primary="อ้างอิง Research",
                aliases=[
                    "อ้างอิง (Research) เอกสาร (Document)",
                    "อ้างอิง (Research)",
                    "อ้างอิง (Research) เอกสาร",
                    "อ้างอิง (Research) เอกสาร(Document)",
                ],
                contains_any=["อ้างอิง", "research", "document"],
            ),
        }

    def _validate_sheet_schema(self, df: pd.DataFrame, sheet_url: str) -> None:
        """
        Fail fast if the loaded sheet doesn't look like our content table.
        This prevents silently ingesting the wrong tab (wrong gid).
        """
        required_tokens = ["หน่วยงาน", "ใบอนุญาต", "ขั้นตอน", "ค่าธรรมเนียม", "ข้อกำหนด", "อ้างอิง"]

        cols = [str(c) for c in df.columns]
        joined = " | ".join(cols)

        hits = sum(1 for t in required_tokens if t in joined)

        if hits < 3:
            raise RuntimeError(
                "Loaded Google Sheet tab does not match expected content schema "
                f"(header tokens matched={hits}/{len(required_tokens)}). "
                "Likely the URL points to the wrong tab (wrong gid) or the sheet format changed.\n"
                f"Given URL: {sheet_url}\n"
                f"Detected columns: {cols}"
            )

    def _get_row_value(self, row: pd.Series, col_name: Optional[str]) -> Optional[str]:
        if not col_name:
            return None
        return self.to_json_safe(row.get(col_name))

    # -----------------------------
    # RAG: content shaping (NEW)
    # -----------------------------
    def _join_nonempty(self, parts: List[str]) -> str:
        parts2 = [p.strip() for p in (parts or []) if p and str(p).strip()]
        return "\n".join(parts2).strip()

    def _build_page_content(self, md: Dict[str, Optional[str]]) -> str:
        """
        Build high-signal embedding text only.
        Keep it compact, procedural, and anchored on what users ask for.
        """
        # Headline/context (short)
        head = self._join_nonempty(
            [
                f"หน่วยงาน: {md.get('department')}" if md.get("department") else "",
                f"ใบอนุญาต: {md.get('license_type')}" if md.get("license_type") else "",
                f"หัวข้อ: {md.get('operation_topic')}" if md.get("operation_topic") else "",
                f"ประเภทการจดทะเบียน: {md.get('registration_type')}" if md.get("registration_type") else "",
            ]
        )

        # High-signal procedural/legal fields
        body = self._join_nonempty(
            [
                f"ขั้นตอน: {md.get('operation_steps')}" if md.get("operation_steps") else "",
                f"เอกสาร: {md.get('identification_documents')}" if md.get("identification_documents") else "",
                f"ค่าธรรมเนียม: {md.get('fees')}" if md.get("fees") else "",
                f"ระยะเวลา: {md.get('operation_duration')}" if md.get("operation_duration") else "",
                f"ช่องทาง/หน่วยงานติดต่อ: {md.get('service_channel')}" if md.get("service_channel") else "",
                f"เงื่อนไข: {md.get('terms_and_conditions')}" if md.get("terms_and_conditions") else "",
                f"ข้อกฎหมาย/ข้อบังคับ: {md.get('legal_regulatory')}" if md.get("legal_regulatory") else "",
                f"ฟอร์ม/เอกสาร: {md.get('restaurant_ai_document')}" if md.get("restaurant_ai_document") else "",
            ]
        )

        extra = ""
        if self.include_research_reference_in_content and md.get("research_reference"):
            extra = f"อ้างอิง: {md.get('research_reference')}"

        text = self._join_nonempty([head, body, extra])

        # Final clamp to avoid huge embeddings
        if self.page_content_max_chars and len(text) > self.page_content_max_chars:
            text = text[: self.page_content_max_chars].rstrip()

        return text.strip()

    def _process_dataframe(self, df: pd.DataFrame, source: str) -> int:
        docs_before = len(self.documents)

        colmap = self._build_column_map(df)

        for idx, row in df.iterrows():
            dept = self._get_row_value(row, colmap.get("department"))
            if dept:
                self.departments_found.add(dept)

            metadata = {
                "row_id": int(idx),
                "department": self._get_row_value(row, colmap.get("department")),
                "license_type": self._get_row_value(row, colmap.get("license_type")),
                "operation_by_department": self._get_row_value(row, colmap.get("operation_by_department")),
                "operation_topic": self._get_row_value(row, colmap.get("operation_topic")),
                "registration_type": self._get_row_value(row, colmap.get("registration_type")),
                "terms_and_conditions": self._get_row_value(row, colmap.get("terms_and_conditions")),
                "service_channel": self._get_row_value(row, colmap.get("service_channel")),
                "operation_steps": self._get_row_value(row, colmap.get("operation_steps")),
                "identification_documents": self._get_row_value(row, colmap.get("identification_documents")),
                "operation_duration": self._get_row_value(row, colmap.get("operation_duration")),
                "fees": self._get_row_value(row, colmap.get("fees")),
                "legal_regulatory": self._get_row_value(row, colmap.get("legal_regulatory")),
                "restaurant_ai_document": self._get_row_value(row, colmap.get("restaurant_ai_document")),
                "research_reference": self._get_row_value(row, colmap.get("research_reference")),
                "source": source,
            }

            # NEW: page_content is NOT "all columns" anymore (reduce embedding noise)
            page_content = self._build_page_content(metadata)

            # If everything is empty (rare), keep a minimal anchor to avoid empty doc
            if not page_content:
                page_content = self._join_nonempty(
                    [
                        f"หน่วยงาน: {metadata.get('department')}" if metadata.get("department") else "",
                        f"ใบอนุญาต: {metadata.get('license_type')}" if metadata.get("license_type") else "",
                        f"หัวข้อ: {metadata.get('operation_topic')}" if metadata.get("operation_topic") else "",
                    ]
                ).strip() or "ข้อมูลกฎหมาย/ขั้นตอนร้านอาหาร (ไม่มีรายละเอียดในแถวนี้)"

            self.documents.append(Document(page_content=page_content, metadata=metadata))

        return len(self.documents) - docs_before

    def get_statistics(self):
        print("\n--- Data Statistics ---")
        print(f"Total documents: {len(self.documents)}")
        print(f"Departments: {len(self.departments_found)}")

        for dept in sorted(self.departments_found):
            count = sum(1 for d in self.documents if d.metadata.get("department") == dept)
            print(f" • {dept}: {count} docs")

        return {
            "total_docs": len(self.documents),
            "departments": list(self.departments_found),
        }