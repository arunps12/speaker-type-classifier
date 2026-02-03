import sys
import traceback
from dataclasses import dataclass
from typing import Optional


@dataclass
class ErrorContext:
    file_name: str
    line_no: int
    func_name: str


def _extract_context(exc_tb) -> ErrorContext:
    """
    Extract last traceback frame info: file, line, function.
    """
    tb_list = traceback.extract_tb(exc_tb)
    if not tb_list:
        return ErrorContext(file_name="unknown", line_no=-1, func_name="unknown")

    last = tb_list[-1]
    return ErrorContext(file_name=last.filename, line_no=last.lineno, func_name=last.name)


class SpeakerTypeClassifierException(Exception):
    """
    Custom project exception that includes source location context.

    Usage:
        try:
            ...
        except Exception as e:
            raise SpeakerTypeClassifierException(e, sys) from e
    """

    def __init__(self, error: Exception, sys_module=sys, message: Optional[str] = None):
        self.original_error = error

        exc_type, exc_value, exc_tb = sys_module.exc_info()
        ctx = _extract_context(exc_tb) if exc_tb else ErrorContext("unknown", -1, "unknown")

        base_msg = message or str(error)
        self.error_message = (
            f"{base_msg} | "
            f"file={ctx.file_name} line={ctx.line_no} func={ctx.func_name} | "
            f"error_type={type(error).__name__}"
        )
        super().__init__(self.error_message)
