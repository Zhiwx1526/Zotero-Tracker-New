"""与 zotero-arxiv-daily 一致的 glob 风格路径匹配。"""

import glob
import re


def glob_match(path: str, pattern: str) -> bool:
    re_pattern = glob.translate(pattern, recursive=True)
    return re.match(re_pattern, path) is not None
