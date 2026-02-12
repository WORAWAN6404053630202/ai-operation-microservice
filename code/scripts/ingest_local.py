# code/scripts/ingest_local.py
from service.data_loader import DataLoader
from service.local_vector_store import ingest_documents

SHEET_URL = "https://docs.google.com/spreadsheets/d/1YnLKV7gJXCu7jvcH1sUL9crMlBCJKOpQfp2wtulMszE/edit?gid=657201027#gid=657201027"


def main():
    dl = DataLoader(config={})
    dl.load_from_google_sheet(SHEET_URL, source_name="google_sheet")
    docs = dl.documents

    print(f"[Ingest] documents={len(docs)}")
    ingest_documents(docs)
    print("[Ingest] Done. You can now run `python main.py`.")


if __name__ == "__main__":
    main()
