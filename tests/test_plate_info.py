from modules.plate_info import decode_plate


def test_decodes_state_and_rto_district():
    v = decode_plate("MH02DL4596")
    assert v["recognized"] is True
    assert v["state"] == "Maharashtra"
    assert v["rto_code"] == "02"
    assert v["rto"] == "Mumbai (West)"
    assert v["series"] == "DL"
    assert v["number"] == "4596"


def test_pads_single_digit_rto_code():
    # "DL7..." -> RTO district 07
    v = decode_plate("DL7CE1111")
    assert v["state"] == "Delhi"
    assert v["rto_code"] == "07"


def test_known_state_unknown_district_still_recognized():
    # A valid state code with a district we don't have a name for still
    # decodes the state, just without a district label.
    v = decode_plate("MH99XY1234")
    assert v["recognized"] is True
    assert v["state"] == "Maharashtra"
    assert v["rto_code"] == "99"
    assert v["rto"] is None


def test_normalizes_spacing_and_case():
    assert decode_plate("  mh 02 dl 4596 ")["plate"] == "MH02DL4596"


def test_bharat_series():
    v = decode_plate("22BH1234AA")
    assert v["recognized"] is True
    assert v["state_code"] == "BH"
    assert "2022" in v["extra"]


def test_unrecognized_plate_format():
    # OCR-garbled / non-standard strings decode to not-recognized rather
    # than inventing a state.
    for bad in ["IZR40294", "XX99YY0000", "", "1234"]:
        assert decode_plate(bad)["recognized"] is False
