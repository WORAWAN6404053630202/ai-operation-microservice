# code/adapter/error/error.py
class CustomError(Exception):
    def __init__(self, status_code: int, detail: str, error_code: str = "UNKNOWN_ERROR"):
        self.status_code = status_code
        self.detail = detail
        self.error_code = error_code
        super().__init__(detail)