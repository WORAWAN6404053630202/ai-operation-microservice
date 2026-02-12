# code/adapter/validate/validate.py
ALLOWED_EXTENSIONS = {"txt", "csv", "json"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS