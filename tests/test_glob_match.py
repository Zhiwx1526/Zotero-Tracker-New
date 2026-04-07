from zotero_tracker.utils_glob import glob_match


def test_glob_match_simple():
    assert glob_match("a/b/c", "a/**")
    assert glob_match("2026/survey/foo", "2026/survey/**")
    assert not glob_match("other/x", "2026/survey/**")
