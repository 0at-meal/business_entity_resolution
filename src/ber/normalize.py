"""Normalization v0 (T04 baseline). T06 will replace/extend this."""
import polars as pl

# Legal forms, prefixes and filler words dropped from the name "core" (US, India, France).
DROP_TOKENS = [
    # India
    "private", "pvt", "limited", "ltd", "llp", "public", "opc",
    # US
    "llc", "inc", "incorporated", "corp", "corporation", "co", "company", "pc", "pllc",
    "lp", "plc", "llc",
    # France
    "sarl", "sas", "sasu", "sa", "eurl", "snc", "sci",
    # prefixes / fillers
    "ms", "dr", "the", "and", "et", "com", "www",
]


def clean_name(e):
    return (e.str.normalize("NFKD")
             .str.replace_all(r"(\p{Latin})\p{Mn}+", "$1")      # strip Latin accents only
             .str.to_lowercase()
             .str.replace_all(r"\bm/s\b", " ")
             .str.replace_all(r"\.com\b|\bwww\.", " ")
             .str.replace_all("&", " and ")
             .str.replace_all(r"\.", "")                         # P.C. -> pc, L.L.C. -> llc
             .str.replace_all(r"[^\p{L}\p{M}\p{N}]+", " ")
             .str.strip_chars())


def name_core_tokens(e):
    return (clean_name(e).str.split(" ")
            .list.eval(pl.element().filter((pl.element() != "") & ~pl.element().is_in(DROP_TOKENS))))


def addr_numbers(e):
    """Unique digit groups in the address with leading zeros stripped."""
    return (e.str.extract_all(r"\d+")
             .list.eval(pl.element().str.strip_chars_start("0"))
             .list.eval(pl.element().filter(pl.element() != ""))
             .list.unique())
