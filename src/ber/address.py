"""Address normalization v1: country-aware abbreviations, state canonicalisation, filler removal.

Output columns per record (see add_address):
    state     canonical state/region code (String, null if not found)
    addr_toks sorted unique informative tokens (no numbers, states, fillers, single letters, native script)
    addr_str  cleaned full address string incl. numbers (for char n-gram features)
Unknown countries get generic cleaning only (country is an open set).
"""
import polars as pl

from .normalize import NONLATIN_RE

B = r"(?-u:\b)"

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "district of columbia": "dc",
    "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id", "illinois": "il",
    "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
    "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc", "south dakota": "sd",
    "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va",
    "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "puerto rico": "pr",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "chattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml",
    "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "tamilnadu": "tn", "telangana": "ts",
    "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk", "uttaranchal": "uk",
    "west bengal": "wb", "delhi": "dl", "new delhi": "dl", "jammu and kashmir": "jk",
    "jammu kashmir": "jk", "puducherry": "py", "pondicherry": "py", "chandigarh": "ch",
    "ladakh": "la", "andaman and nicobar islands": "an", "lakshadweep": "ld",
    "dadra and nagar haveli": "dn", "daman and diu": "dd",
}
IN_CODE_ALIASES = {"or": "od", "ct": "cg", "tg": "ts", "ut": "uk", "del": "dl"}
FR_REGIONS = {
    "hauts de france": "hdf", "nord": "hdf", "pas de calais": "hdf",
    "nouvelle aquitaine": "naq", "gironde": "naq",
    "pays de la loire": "pdl", "loire atlantique": "pdl",
}

US_ABBR = {
    "st": "street", "str": "street", "rd": "road", "dr": "drive", "drv": "drive", "ave": "avenue",
    "av": "avenue", "avn": "avenue", "ln": "lane", "ct": "court", "crt": "court",
    "blvd": "boulevard", "hwy": "highway", "pkwy": "parkway", "cir": "circle", "pl": "place",
    "ter": "terrace", "trl": "trail", "sq": "square", "mt": "mount", "ft": "fort", "n": "north",
    "s": "south", "e": "east", "w": "west", "ne": "northeast", "nw": "northwest",
    "se": "southeast", "sw": "southwest", "hts": "heights", "twp": "township", "cty": "county",
    "expy": "expressway", "fwy": "freeway", "jct": "junction", "rte": "route",
}
IN_ABBR = {
    "rd": "road", "st": "street", "ngr": "nagar", "opp": "opposite", "nr": "near",
    "bldg": "building", "flr": "floor", "apt": "apartment", "mkt": "market", "sec": "sector",
    "sect": "sector", "extn": "extension", "ext": "extension", "clny": "colony", "chwk": "chowk",
}
FR_ABBR = {
    "r": "rue", "av": "avenue", "ave": "avenue", "bd": "boulevard", "blvd": "boulevard",
    "bld": "boulevard", "all": "allee", "imp": "impasse", "pl": "place", "rte": "route",
    "ch": "chemin", "chem": "chemin", "st": "saint", "ste": "sainte", "fbg": "faubourg",
    "sq": "square", "crs": "cours", "qu": "quai", "res": "residence", "lot": "lotissement",
}

COMMON_STOP = {"na", "null", "none", "unknown", "address", "and", "of", "the"}
US_STOP = COMMON_STOP | {"unit", "suite", "ste", "apartment", "apt", "floor", "fl", "building",
                         "bldg", "usa", "us", "box", "po", "pmb", "number", "no", "rm", "room"}
IN_STOP = COMMON_STOP | {"no", "near", "opposite", "floor", "flat", "plot", "door", "house", "shop",
                         "office", "block", "sector", "at", "po", "ps", "dist", "district", "city",
                         "india", "ground", "first", "second", "third", "unit", "room", "c", "o",
                         "h", "tal", "taluk", "tehsil", "via", "post", "building", "apartment"}
FR_STOP = COMMON_STOP | {"de", "la", "le", "les", "du", "des", "d", "l", "et", "bis", "ter",
                         "france", "cedex", "n", "no", "au", "aux", "en"}

COUNTRY_CFG = {
    # prefer: which signal wins when both a state name and a state code are present
    "US": dict(names=US_STATES, codes=set(US_STATES.values()), aliases={}, abbr=US_ABBR, stop=US_STOP,
               prefer="code"),    # 'Fort Washington, MD' -> md, not wa
    "India": dict(names=IN_STATES, codes=set(IN_STATES.values()) | set(IN_CODE_ALIASES),
                  aliases=IN_CODE_ALIASES, abbr=IN_ABBR, stop=IN_STOP,
                  prefer="name"),  # 'ka' is often Hindi, not Karnataka
    "France": dict(names=FR_REGIONS, codes=set(), aliases={}, abbr=FR_ABBR, stop=FR_STOP, prefer="name"),
}
GENERIC_CFG = dict(names={}, codes=set(), aliases={}, abbr={}, stop=COMMON_STOP, prefer="name")


def clean_addr_base(e):
    return (e.str.normalize("NFKD")
             .str.replace_all(r"(\p{Latin})\p{Mn}+", "$1")
             .str.replace_all("[\u200c\u200d]", "")
             .str.to_lowercase()
             .str.replace_all(rf"{B}(?:n/a|c/o){B}", " ")
             .str.replace_all(r"[^\p{L}\p{M}\p{N}]+", " ")
             .str.strip_chars())


def _alternation(names):
    ordered = sorted(names, key=len, reverse=True)   # longest first: 'west virginia' before 'virginia'
    return B + "(" + "|".join(ordered) + ")" + B


def add_address(df, country, col="business_address"):
    """df: rows of ONE country with a raw address column. Returns df + state, addr_toks, addr_str."""
    cfg = COUNTRY_CFG.get(country, GENERIC_CFG)
    base = clean_addr_base(pl.col(col))
    df = df.with_columns(_base=base)

    if cfg["names"]:
        alt = _alternation(cfg["names"])
        state_from_name = (pl.col("_base").str.extract_all(alt).list.last()
                           .replace_strict(cfg["names"], default=None, return_dtype=pl.String))
        stripped = pl.col("_base").str.replace_all(alt, " ")
    else:
        state_from_name = pl.lit(None, dtype=pl.String)
        stripped = pl.col("_base")
    df = df.with_columns(_state_name=state_from_name,
                         _raw_toks=stripped.str.split(" ")
                                           .list.eval(pl.element().filter(pl.element() != "")))

    if cfg["codes"]:
        code_tok = (pl.col("_raw_toks").list.eval(pl.element().filter(pl.element().is_in(list(cfg["codes"]))))
                    .list.last())
        if cfg["aliases"]:
            code_tok = code_tok.replace(cfg["aliases"])
    else:
        code_tok = pl.lit(None, dtype=pl.String)

    abbr, stop, codes = cfg["abbr"], list(cfg["stop"]), list(cfg["codes"])
    el = pl.element()
    mapped = pl.col("_raw_toks").list.eval(el.replace(abbr)) if abbr else pl.col("_raw_toks")
    state = (pl.coalesce(code_tok, "_state_name") if cfg["prefer"] == "code"
             else pl.coalesce("_state_name", code_tok))
    df = df.with_columns(state=state, _toks=mapped)
    keep = (~el.is_in(stop) & ~el.str.contains(r"^\d+$")
            & (el.str.len_chars() > 1) & ~el.str.contains(NONLATIN_RE))
    if codes:
        keep = keep & ~el.is_in(codes)
    df = df.with_columns(
        addr_toks=pl.col("_toks").list.eval(el.filter(keep)).list.unique().list.sort(),
        addr_str=pl.col("_toks").list.join(" "),
        addr_native=pl.col("_toks").list.eval(el.filter(el.str.contains(NONLATIN_RE))).list.unique(),
    )
    return df.drop("_base", "_state_name", "_raw_toks", "_toks")


def learn_native_state(recs, pairs, min_support=20, min_share=0.9):
    """Learn native-script address token -> state code from true pairs (candidate state unknown,
    S1 state known). recs: id, src, state, addr_native. pairs: (s1, c)."""
    a = (recs.filter((pl.col("src") == 1) & pl.col("state").is_not_null())
             .select(pl.col("id").alias("s1"), pl.col("state").alias("s1_state")))
    b = (recs.filter(pl.col("state").is_null() & (pl.col("addr_native").list.len() > 0))
             .select(pl.col("id").alias("c"), "addr_native"))
    ex = (pairs.join(b, on="c").join(a, on="s1")
               .explode("addr_native", empty_as_null=True).drop_nulls("addr_native"))
    counts = ex.group_by("addr_native", "s1_state").agg(pl.len().alias("n"))
    return (counts.group_by("addr_native")
                  .agg(pl.col("s1_state").sort_by("n", descending=True).first().alias("state"),
                       pl.col("n").max().alias("support"), pl.col("n").sum().alias("total"))
                  .with_columns(share=pl.col("support") / pl.col("total"))
                  .filter((pl.col("support") >= min_support) & (pl.col("share") >= min_share))
                  .select(pl.col("addr_native").alias("token"), "state", "support", "share")
                  .sort("support", descending=True))


def apply_native_state(recs, m):
    """Fill missing states from native-script address tokens using a learned map."""
    ex = (recs.filter(pl.col("state").is_null() & (pl.col("addr_native").list.len() > 0))
              .select("id", "addr_native")
              .explode("addr_native", empty_as_null=True)
              .join(m.select(pl.col("token").alias("addr_native"), pl.col("state").alias("st")),
                    on="addr_native", how="inner"))
    found = ex.group_by("id").agg(pl.col("st").mode().first().alias("st_native"))
    return (recs.join(found, on="id", how="left")
                .with_columns(state=pl.coalesce("state", "st_native"))
                .drop("st_native"))
