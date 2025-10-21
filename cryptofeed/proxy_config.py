"""Proxy configuration data models."""
from __future__ import annotations

from typing import Dict, List, Optional, Literal, Mapping
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator, ConfigDict, model_validator
from pydantic_settings import BaseSettings


_ALLOWED_PROXY_SCHEMES = {"http", "https", "socks4", "socks4a", "socks5", "socks5h"}


def _validate_proxy_scheme(scheme: str) -> str:
    scheme_lower = scheme.lower()
    if scheme_lower in _ALLOWED_PROXY_SCHEMES:
        return scheme_lower
    raise ValueError(f"Unsupported proxy scheme: {scheme}")


class ProxyUrlConfig(BaseModel):
    """Individual proxy URL configuration within pools."""
    model_config = ConfigDict(frozen=True, extra='forbid')

    url: str = Field(..., description="Proxy URL (e.g., socks5://user:pass@host:1080)")
    weight: float = Field(default=1.0, ge=0.1, le=10.0, description="Proxy weight for selection")
    enabled: bool = Field(default=True, description="Whether proxy is enabled")

    @field_validator('url')
    @classmethod
    def validate_proxy_url(cls, v: str) -> str:
        parsed = urlparse(v)
        if '://' not in v:
            raise ValueError("Proxy URL must include scheme")
        if not parsed.scheme:
            raise ValueError("Proxy URL must include scheme")
        _validate_proxy_scheme(parsed.scheme)
        if not parsed.hostname:
            raise ValueError("Proxy URL must include hostname")
        if not parsed.port:
            raise ValueError("Proxy URL must include port")
        return v

    @property
    def scheme(self) -> str:
        return urlparse(self.url).scheme

    @property
    def host(self) -> str:
        return urlparse(self.url).hostname

    @property
    def port(self) -> int:
        return urlparse(self.url).port


class ProxyPoolConfig(BaseModel):
    """Proxy pool configuration with multiple proxies and selection strategy."""
    model_config = ConfigDict(extra='forbid')

    proxies: List[ProxyUrlConfig] = Field(..., min_length=1, description="List of proxy configurations")
    strategy: Literal['round_robin', 'random', 'least_connections'] = Field(
        default='round_robin',
        description="Proxy selection strategy"
    )


class ProxyConfig(BaseModel):
    """Single proxy configuration with URL validation, extended to support pools."""
    model_config = ConfigDict(frozen=True, extra='forbid')

    url: Optional[str] = Field(default=None, description="Proxy URL (e.g., socks5://user:pass@host:1080)")
    timeout_seconds: int = Field(default=30, ge=1, le=300)
    pool: Optional[ProxyPoolConfig] = Field(default=None, description="Proxy pool configuration")

    @model_validator(mode='before')
    @classmethod
    def _coerce_str(cls, value):
        if isinstance(value, str):
            return {'url': value}
        return value

    @field_validator('url')
    @classmethod
    def validate_proxy_url(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        parsed = urlparse(v)
        if '://' not in v:
            raise ValueError("Proxy URL must include scheme (http, socks5, socks4)")
        if not parsed.scheme:
            raise ValueError("Proxy URL must include scheme (http, socks5, socks4)")
        _validate_proxy_scheme(parsed.scheme)
        if not parsed.hostname:
            raise ValueError("Proxy URL must include hostname")
        if not parsed.port:
            raise ValueError("Proxy URL must include port")
        return v

    @property
    def scheme(self) -> Optional[str]:
        if self.url:
            return urlparse(self.url).scheme
        return None

    @property
    def host(self) -> Optional[str]:
        if self.url:
            return urlparse(self.url).hostname
        return None

    @property
    def port(self) -> Optional[int]:
        if self.url:
            return urlparse(self.url).port
        return None


class ConnectionProxies(BaseModel):
    model_config = ConfigDict(extra='forbid')

    @model_validator(mode='before')
    @classmethod
    def _coerce_aliases(cls, data):
        if isinstance(data, str):
            return {'http': data}
        if isinstance(data, Mapping):
            data = dict(data)
            rest_value = data.pop('rest', None)
            if rest_value is not None and 'http' not in data:
                data['http'] = rest_value
            ws_value = data.pop('ws', None)
            if ws_value is not None and 'websocket' not in data:
                data['websocket'] = ws_value
            return data
        return data

    http: Optional[ProxyConfig] = Field(default=None, description="HTTP/REST proxy")
    websocket: Optional[ProxyConfig] = Field(default=None, description="WebSocket proxy")


class ProxySettings(BaseSettings):
    model_config = ConfigDict(
        env_prefix='CRYPTOFEED_PROXY_',
        env_nested_delimiter='__',
        case_sensitive=False,
        extra='forbid'
    )

    enabled: bool = Field(default=False, description="Enable proxy functionality")
    default: Optional[ConnectionProxies] = Field(
        default=None,
        description="Default proxy configuration for all exchanges"
    )
    exchanges: Dict[str, ConnectionProxies] = Field(
        default_factory=dict,
        description="Exchange-specific proxy overrides"
    )

    def model_post_init(self, __context) -> None:
        if self.exchanges:
            self.exchanges = {key.casefold(): value for key, value in self.exchanges.items()}

    def get_proxy(self, exchange_id: str, connection_type: Literal['http', 'websocket']) -> Optional[ProxyConfig]:
        if not self.enabled:
            return None

        key = exchange_id.casefold() if exchange_id else exchange_id
        if key in self.exchanges:
            proxy = getattr(self.exchanges[key], connection_type, None)
            if proxy is not None:
                return proxy

        if self.default:
            return getattr(self.default, connection_type, None)
        return None
