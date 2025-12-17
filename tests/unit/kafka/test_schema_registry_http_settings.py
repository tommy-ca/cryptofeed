from unittest.mock import patch, Mock

from cryptofeed.backends.kafka_schema import (
    SchemaRegistryConfig,
    ConfluentSchemaRegistry,
)


def test_schema_registry_uses_proxy_and_timeout(monkeypatch):
    monkeypatch.setenv("CF_SCHEMA_REGISTRY_TIMEOUT", "7")
    monkeypatch.setenv("CF_SCHEMA_REGISTRY_PROXY", "http://sr-proxy:8080")

    cfg = SchemaRegistryConfig(registry_type="confluent", url="http://sr:8081")
    reg = ConfluentSchemaRegistry(cfg)

    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"id": 1, "version": 1, "schema": '{"type":"record"}'}

    with patch("requests.post", return_value=mock_resp) as mp, patch(
        "requests.get", return_value=mock_resp
    ) as mg, patch("requests.put", return_value=mock_resp) as mput:
        reg.register_schema("trades", '{"type":"record"}', "PROTOBUF")
        reg.get_schema_by_id(1)
        reg.get_schema_by_version("trades", 1)
        reg.check_compatibility("trades", '{"type":"record"}', version=1)
        reg.set_compatibility_mode("trades", "BACKWARD")

    expected_proxies = {"http": "http://sr-proxy:8080", "https": "http://sr-proxy:8080"}
    for call in (*mp.call_args_list, *mg.call_args_list, *mput.call_args_list):
        kwargs = call.kwargs
        assert kwargs["timeout"] == 7.0
        assert kwargs["proxies"] == expected_proxies
