"""Decode an Indian number plate into its registration details.

Indian plates encode their own origin: `MH 02 DL 4596` means state MH
(Maharashtra), RTO district 02 (Mumbai West), series DL, number 4596. That
mapping is entirely public and static, so we can show "where this vehicle is
registered" for every plate without any external API, key, or owner data.

Only the state and RTO district are derivable from the plate — maker, model,
and owner are NOT in the number and would need a (restricted) VAHAN-backed
API, so this module deliberately never claims to know them.
"""

import re

# Two-letter state / union-territory codes used on current Indian plates.
STATE_CODES = {
    "AN": "Andaman & Nicobar Islands",
    "AP": "Andhra Pradesh",
    "AR": "Arunachal Pradesh",
    "AS": "Assam",
    "BR": "Bihar",
    "CH": "Chandigarh",
    "CG": "Chhattisgarh",
    "DD": "Daman & Diu",
    "DL": "Delhi",
    "DN": "Dadra & Nagar Haveli",
    "GA": "Goa",
    "GJ": "Gujarat",
    "HR": "Haryana",
    "HP": "Himachal Pradesh",
    "JK": "Jammu & Kashmir",
    "JH": "Jharkhand",
    "KA": "Karnataka",
    "KL": "Kerala",
    "LA": "Ladakh",
    "LD": "Lakshadweep",
    "MP": "Madhya Pradesh",
    "MH": "Maharashtra",
    "MN": "Manipur",
    "ML": "Meghalaya",
    "MZ": "Mizoram",
    "NL": "Nagaland",
    "OD": "Odisha",
    "OR": "Odisha",  # older code, still on the road
    "PB": "Punjab",
    "PY": "Puducherry",
    "RJ": "Rajasthan",
    "SK": "Sikkim",
    "TN": "Tamil Nadu",
    "TS": "Telangana",
    "TG": "Telangana",  # older code
    "TR": "Tripura",
    "UP": "Uttar Pradesh",
    "UK": "Uttarakhand",
    "UA": "Uttarakhand",  # older code
    "WB": "West Bengal",
}

# RTO-district names, keyed by "STATE+DD". Not exhaustive (India has ~1000+
# RTOs) — a curated set of major offices, with graceful fallback to just the
# code for anything not listed. Covers the demo plates plus common metros.
RTO_DISTRICTS = {
    "MH01": "Mumbai (South)",
    "MH02": "Mumbai (West)",
    "MH03": "Mumbai (East)",
    "MH04": "Thane",
    "MH05": "Kalyan",
    "MH12": "Pune",
    "MH14": "Pimpri-Chinchwad",
    "MH20": "Chhatrapati Sambhajinagar (Aurangabad)",
    "MH31": "Nagpur",
    "MH43": "Navi Mumbai",
    "DL01": "Delhi (Central)",
    "DL02": "Delhi (New Delhi)",
    "DL03": "Delhi (South)",
    "DL07": "Delhi (East)",
    "KA01": "Bengaluru (Koramangala)",
    "KA02": "Bengaluru (Rajajinagar)",
    "KA03": "Bengaluru (Indiranagar)",
    "KA05": "Bengaluru (Jayanagar)",
    "TN01": "Chennai (Central)",
    "TN07": "Chennai (West)",
    "TN09": "Chennai (North)",
    "GJ01": "Ahmedabad",
    "GJ05": "Surat",
    "UP32": "Lucknow",
    "UP16": "Noida (Gautam Buddha Nagar)",
    "HR26": "Gurugram",
    "RJ14": "Jaipur (South)",
    "WB02": "Kolkata (Beltala)",
    "TS09": "Hyderabad (Malakpet)",
    "AP16": "Vijayawada",
    "KL01": "Thiruvananthapuram",
    "KL07": "Kochi (Ernakulam)",
    "PB10": "Ludhiana",
    "MP09": "Indore",
    "MP04": "Bhopal",
}

# Standard format: 2-letter state, 1-2 digit RTO, 1-3 letter series, 1-4 digit
# number, e.g. MH02DL4596, KA05AB1, DL7CE1234.
_STANDARD = re.compile(r"^([A-Z]{2})(\d{1,2})([A-Z]{1,3})(\d{1,4})$")

# Bharat (BH) series — nationwide, not tied to a state: 2-digit year, BH,
# 4-digit number, 1-2 letter series, e.g. 22BH1234AA.
_BH_SERIES = re.compile(r"^(\d{2})BH(\d{4})([A-Z]{1,2})$")


def decode_plate(plate):
    """Return registration details decoded from the plate string.

    Result dict always has a `recognized` bool; when True it carries
    `state`, and (for standard plates) `rto_code`, `rto`, `series`, `number`.
    """
    if not plate:
        return {"recognized": False}

    cleaned = re.sub(r"[^A-Za-z0-9]", "", plate).upper()

    bh = _BH_SERIES.match(cleaned)
    if bh:
        year, number, series = bh.groups()
        return {
            "recognized": True,
            "plate": cleaned,
            "state": "Bharat series (nationwide registration)",
            "state_code": "BH",
            "series": series,
            "number": number,
            "extra": f"Registered 20{year}",
        }

    match = _STANDARD.match(cleaned)
    if not match:
        return {"recognized": False, "plate": cleaned}

    state_code, rto_code, series, number = match.groups()
    state = STATE_CODES.get(state_code)
    if not state:
        # Two leading letters that aren't a known state code — treat as
        # unrecognized rather than inventing a state.
        return {"recognized": False, "plate": cleaned}

    rto_code_padded = rto_code.zfill(2)
    rto = RTO_DISTRICTS.get(state_code + rto_code_padded)

    return {
        "recognized": True,
        "plate": cleaned,
        "state": state,
        "state_code": state_code,
        "rto_code": rto_code_padded,
        "rto": rto,  # may be None if that district isn't in our list
        "series": series,
        "number": number,
    }
