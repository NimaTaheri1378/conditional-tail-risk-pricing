from ctrsdf.analysis.results import ff12_from_sic


def test_ff12_from_sic_core_ranges():
    assert ff12_from_sic(6020) == "Finance"
    assert ff12_from_sic(2834) == "Healthcare"
    assert ff12_from_sic(7372) == "Business Equipment"
    assert ff12_from_sic(None) == "Other"
