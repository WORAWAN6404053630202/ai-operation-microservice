# code/service/data_loader.py
"""
Data Loader Adapter
Loads data from Google Sheets and converts to document format
"""
import re
import math
import pandas as pd
from langchain_core.documents import Document
from typing import List


class DataLoader:
    """
    Flexible Data Loader
    - Load Google Sheet via CSV export URL
    - Normalize multi-line headers
    - Convert NaN/None → None
    - Build Document + metadata compatible with pipeline
    """

    def __init__(self, config):
        self.config = config
        self.documents = []
        self.departments_found = set()

    @staticmethod
    def clean_header(name: str) -> str:
        if not isinstance(name, str):
            return name
        name = name.replace("\n", " ").replace("\r", " ")
        name = re.sub(r"\s+", " ", name)
        return name.strip()

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
        csv_url = sheet_url.replace("/edit?gid=", "/export?format=csv&gid=")
        df = pd.read_csv(csv_url)
        df.rename(columns={c: self.clean_header(c) for c in df.columns}, inplace=True)

        print("Columns detected:")
        for c in df.columns:
            print(" •", repr(c))

        print(f"\nLoaded {len(df)} rows")
        docs_added = self._process_dataframe(df, source=source_name or csv_url)

        print(f"Added {docs_added} documents")
        return docs_added

    def _process_dataframe(self, df: pd.DataFrame, source: str) -> int:
        docs_before = len(self.documents)

        for idx, row in df.iterrows():
            record = {k: ("" if pd.isna(v) else str(v).strip()) for k, v in row.items()}

            content = "\n".join(f"{k}: {v}" for k, v in record.items() if v != "")

            dept = self.to_json_safe(row.get("หน่วยงาน"))
            if dept:
                self.departments_found.add(dept)

            metadata = {
                "row_id": int(idx),
                "department": self.to_json_safe(row.get("หน่วยงาน")),
                "license_type": self.to_json_safe(row.get("ใบอนุญาต")),
                "operation_by_department": self.to_json_safe(row.get("การดำเนินการ ตามหน่วยงาน")),
                "operation_topic": self.to_json_safe(row.get("หัวข้อการดำเนินการ")),
                "registration_type": self.to_json_safe(row.get("ประเภทการจดทะเบียน")),
                "terms_and_conditions": self.to_json_safe(row.get("เงื่อนไขและหลักเกณฑ์")),
                "service_channel": self.to_json_safe(row.get("ช่องทางการ ให้บริการ")),
                "operation_steps": self.to_json_safe(row.get("ขั้นตอนการดำเนินการ")),
                "identification_documents": self.to_json_safe(row.get("เอกสาร ยืนยันตัวตน")),
                "operation_duration": self.to_json_safe(row.get("ระยะเวลา การดำเนินการ")),
                "fees": self.to_json_safe(row.get("ค่าธรรมเนียม")),
                "legal_regulatory": self.to_json_safe(row.get("ข้อกำหนดทางกฎหมาย และข้อบังคับ")),
                "restaurant_ai_document": self.to_json_safe(row.get("เอกสาร AI ร้านอาหาร")),
                "research_reference": self.to_json_safe(row.get("อ้างอิง Research")),
                "source": source,
            }

            self.documents.append(Document(page_content=content, metadata=metadata))

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
