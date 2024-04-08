from pathlib import Path

def get_input_path(str_path: str) -> Path:
    if isinstance(str_path, Path):
        str_path = str(str_path)
    if str_path[:2] in ('r"', 'r"'):
        str_path = str_path[2:-1]
    if str_path[:1] in ('"', "'"):
        str_path = str_path[1:-1]
    str_path = r"{}".format(str_path)
    return Path(str_path)
