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


def addr_numbers_int(e):
    """Unique digit groups as UInt32 (leading zeros vanish; very long runs keep their last 9 digits)."""
    return (e.str.extract_all(r"\d+")
             .list.eval(pl.element().str.slice(-9).cast(pl.UInt32, strict=False))
             .list.eval(pl.element().filter(pl.element() > 0))
             .list.unique())


# ============================================================================ v1 (T06)
NONLATIN_RE = r"[^\p{Latin}\p{Common}\p{Inherited}]"

LEGAL_V1 = [
    # India
    "private", "pvt", "limited", "ltd", "llp", "opc", "public",
    # US
    "llc", "inc", "incorporated", "corp", "corporation", "co", "company", "pc", "pllc",
    "lp", "plc", "lc",
    # France
    "sarl", "sas", "sasu", "sa", "eurl", "snc", "sci", "ei", "selarl", "scp", "scop",
]
PREFIX_FILLER_V1 = ["ms", "dr", "mr", "mrs", "messrs", "the", "and", "et", "und"]
GENERIC_V1 = ["center", "centre", "services", "service", "partners", "group", "groupe",
              "holding", "holdings", "international", "developpement", "development",
              "distribution", "participations"]


def is_nonlatin(e):
    return e.str.contains(NONLATIN_RE)


def clean_name_v1(e):
    """Lowercased, accent-stripped (Latin only), punctuation-free name string."""
    return (e.str.normalize("NFKD")
             .str.replace_all(r"(\p{Latin})\p{Mn}+", "$1")
             .str.to_lowercase()
             .str.replace_all(r"^.*?(?-u:\b)(?:formerly|fka|f/k/a)(?-u:\b)[:\s]*", "")   # keep the former name
             .str.replace_all(r"\s(?-u:\b)(?:dba|d/b/a|aka|a/k/a|t/a)(?-u:\b).*$", "")   # keep the legal name
             .str.replace_all(r"(?-u:\b)m/s(?-u:\b)", " ")
             .str.replace_all(r"https?://|(?-u:\b)www\.", " ")
             .str.replace_all(r"\.(?:com|net|org|co\.in|in|fr|biz|info|us)(?-u:\b)", " ")
             .str.replace_all(r"[&+@]", " ")
             .str.replace_all(r"\.", "")                                     # S.A.R.L. -> sarl
             .str.replace_all(r"[^\p{L}\p{M}\p{N}]+", " ")
             .str.replace_all(r"(?-u:\b)s ?a ?r ?l(?-u:\b)", "sarl")                      # spaced-out legal forms
             .str.replace_all(r"(?-u:\b)s ?a ?s ?u(?-u:\b)", "sasu")
             .str.replace_all(r"(?-u:\b)s ?a ?s(?-u:\b)", "sas")
             .str.replace_all(r"(?-u:\b)e ?u ?r ?l(?-u:\b)", "eurl")
             .str.replace_all(r"(?-u:\b)p ?l ?l ?c(?-u:\b)", "pllc")
             .str.replace_all(r"(?-u:\b)l ?l ?c(?-u:\b)", "llc")
             .str.replace_all(r"(?-u:\b)l ?l ?p(?-u:\b)", "llp")
             .str.replace_all(r"(?-u:\b)p ?c(?-u:\b)", "pc")
             .str.strip_chars())


def name_tokens_v1(e):
    """Cleaned name -> token list (legal words kept; used for native-script alignment)."""
    return clean_name_v1(e).str.split(" ").list.eval(pl.element().filter(pl.element() != ""))


def core_from_tokens_v1(toks):
    """Drop legal forms, prefixes/fillers, and generic words that trail after a legal form."""
    el = pl.element()
    legal = el.is_in(LEGAL_V1)
    after_legal = legal.cast(pl.UInt8).cum_max() > 0
    trailing_generic = el.is_in(GENERIC_V1).cast(pl.UInt8).reverse().cum_min().reverse() == 1
    return toks.list.eval(el.filter(~legal & ~el.is_in(PREFIX_FILLER_V1)
                                    & ~(after_legal & trailing_generic)))


def learn_native_dict(toks, pairs, min_support=2, min_share=0.5):
    """Learn native-script token -> Latin token from true pairs with equal token counts.
    toks: frame(id, toks list, nonlatin bool). pairs: frame(s1, c) of true matches."""
    a = toks.select(pl.col("id").alias("s1"), pl.col("toks").alias("a_t"))
    b = toks.filter(pl.col("nonlatin")).select(pl.col("id").alias("c"), pl.col("toks").alias("b_t"))
    P = (pairs.join(b, on="c").join(a, on="s1")
              .filter((pl.col("a_t").list.len() == pl.col("b_t").list.len())
                      & (pl.col("a_t").list.len() > 0))
              .select("a_t", "b_t"))
    ex = P.explode(["a_t", "b_t"]).filter(pl.col("b_t").str.contains(NONLATIN_RE))
    counts = ex.group_by("b_t", "a_t").agg(pl.len().alias("n"))
    d = (counts.group_by("b_t")
               .agg(pl.col("a_t").sort_by("n", descending=True).first().alias("latin"),
                    pl.col("n").max().alias("support"),
                    pl.col("n").sum().alias("total"))
               .with_columns(share=pl.col("support") / pl.col("total"))
               .filter((pl.col("support") >= min_support) & (pl.col("share") >= min_share))
               .select(pl.col("b_t").alias("native"), "latin", "support", "share")
               .sort("support", descending=True))
    return d, P.height


def apply_native_dict(toks_df, d):
    """Replace native tokens with learned Latin tokens. Returns (frame with toks_nat, coverage stats)."""
    nat = toks_df.filter(pl.col("nonlatin")).select("id", "toks")
    ex = (nat.explode("toks", empty_as_null=True)
             .with_columns(pos=pl.int_range(pl.len()).over("id"))
             .join(d.select("native", "latin"), left_on="toks", right_on="native", how="left"))
    native_tok = ex.filter(pl.col("toks").str.contains(NONLATIN_RE))
    stats = {"native_tokens": native_tok.height,
             "mapped_share": float(native_tok["latin"].is_not_null().mean()) if native_tok.height else None,
             "native_names": nat.height}
    back = (ex.with_columns(t=pl.coalesce("latin", "toks"))
              .sort("id", "pos")
              .group_by("id", maintain_order=True)
              .agg(pl.col("t").drop_nulls()))
    out = (toks_df.join(back, on="id", how="left")
                  .with_columns(toks_nat=pl.coalesce("t", "toks"))
                  .drop("t"))
    return out, stats


def num_ext(e):
    """Number list extended with leading-digit-dropped variants (257 -> 57, 208 -> 8)."""
    trunc = e.list.eval(
        pl.element() % pl.lit(10.0).pow((pl.element().cast(pl.Float64).log10() + 1e-9).floor())
                         .cast(pl.UInt32))
    return (pl.concat_list([e, trunc])
              .list.eval(pl.element().filter(pl.element() > 0))
              .list.unique())
