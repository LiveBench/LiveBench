import logging
from pathlib import Path

from rich.highlighter import NullHighlighter
from rich.logging import RichHandler


def _setup_root_logger() -> None:
    logger = logging.getLogger("minisweagent")
    logger.setLevel(logging.DEBUG)
    # No highlighter: ReprHighlighter livelocks on pathological messages
    # while holding the global logging lock, freezing all batch workers.
    _handler = RichHandler(
        show_path=False,
        show_time=False,
        show_level=False,
        markup=True,
        highlighter=NullHighlighter(),
    )
    _formatter = logging.Formatter("%(name)s: %(levelname)s: %(message)s")
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)


def add_file_handler(path: Path | str, level: int = logging.DEBUG, *, print_path: bool = True) -> None:
    logger = logging.getLogger("minisweagent")
    handler = logging.FileHandler(path)
    handler.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if print_path:
        print(f"Logging to '{path}'")


_setup_root_logger()
logger = logging.getLogger("minisweagent")


__all__ = ["logger"]
