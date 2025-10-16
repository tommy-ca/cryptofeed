# Funding Schema Field Mapping

| Canonical Field (Protobuf) | Cryptofeed Dataclass | tardis-node JSON | DBN Layout | Notes |
| --- | --- | --- | --- | --- |
| `exchange` | `Funding.exchange` | `exchange` | `exchange` | |
| `symbol` | `Funding.symbol` | `instrument` | `symbol` | |
| `mark_price` | `Funding.mark_price` (`Decimal`) | `markPrice` (`string`) | `mark_price` (int64, scale 1e-8) | Canonical scale defined by Cryptofeed |
| `rate` | `Funding.rate` (`Decimal`) | `fundingRate` (`string`) | `rate` (int64, scale 1e-8) | |
| `predicted_rate` | `Funding.predicted_rate` | `predictedFundingRate` | reserved | Optional |
| `next_funding_time` | `Funding.next_funding_time` (float s) | `nextFundingTime` (µs) | `next_funding_ts` (µs) | Convert seconds to microseconds |
| `timestamp` | `Funding.timestamp` (float s) | `ts_event` (µs) | `timestamp` (µs) | |
