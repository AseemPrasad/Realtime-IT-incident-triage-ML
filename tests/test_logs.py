import pandas as pd

from incident_triage.logs import derive_ground_truth, extract_json_object, heuristic_log_analysis


def test_derive_ground_truth_prefers_failure_signals():
    row = pd.Series(
        {
            "Level": "ERROR",
            "Component": "dfs.DataNode$PacketResponder",
            "Content": "PacketResponder 1 terminating due to socket exception",
        }
    )
    target = derive_ground_truth(row)
    assert target["severity"] in {"HIGH", "CRITICAL"}
    assert "HDFS" in target["error_domain"]


def test_extract_json_object_handles_noise():
    text = "prefix {\"error_domain\": \"A\", \"severity\": \"HIGH\", \"probable_cause\": \"B\"} suffix"
    parsed = extract_json_object(text)
    assert parsed["severity"] == "HIGH"


def test_heuristic_log_analysis_returns_model():
    analysis = heuristic_log_analysis("BLOCK* NameSystem.addStoredBlock: blockMap updated")
    assert analysis.severity == "INFO"

