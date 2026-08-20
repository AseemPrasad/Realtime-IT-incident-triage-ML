from incident_triage.telemetry import TelemetryFeatureExtractor


def test_telemetry_feature_extraction_shapes():
    extractor = TelemetryFeatureExtractor()
    frame = extractor.extract_from_values([1, 2, 3, 4, 5, 6, 7])
    assert frame.shape[0] == 1
    assert set(extractor.feature_names()).issubset(frame.columns)


def test_telemetry_heuristic_risk_moves_up_on_spike():
    extractor = TelemetryFeatureExtractor()
    healthy = extractor.heuristic_risk([32, 33, 31, 32, 33, 32, 31])
    spiky = extractor.heuristic_risk([32, 33, 31, 32, 33, 32, 97])
    assert spiky > healthy

