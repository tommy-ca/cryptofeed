"""Cryptofeed Backends"""
from cryptofeed.backends.nats import (
    TradeNATS,
    FundingNATS,
    BookNATS,
    TickerNATS,
    OpenInterestNATS,
    LiquidationsNATS,
    CandlesNATS,
    OrderInfoNATS,
    TransactionsNATS,
    BalancesNATS,
    FillsNATS
)
from cryptofeed.backends.socket import UDPCallback, HTTPCallback, SocketCallback
from cryptofeed.backends.tcp import TCPCallback
from cryptofeed.backends.kafka import (
    TradeKafka,
    FundingKafka,
    BookKafka,
    TickerKafka,
    OpenInterestKafka,
    LiquidationsKafka,
    CandlesKafka,
    OrderInfoKafka,
    TransactionsKafka,
    BalancesKafka,
    FillsKafka
)
from cryptofeed.backends.rabbit import (
    TradeRabbit,
    FundingRabbit,
    BookRabbit,
    TickerRabbit,
    OpenInterestRabbit,
    LiquidationsRabbit,
    CandlesRabbit,
    OrderInfoRabbit,
    TransactionsRabbit,
    BalancesRabbit,
    FillsRabbit
)
from cryptofeed.backends.redis import (
    TradeRedis,
    FundingRedis,
    BookRedis,
    TickerRedis,
    OpenInterestRedis,
    LiquidationsRedis,
    CandlesRedis,
    OrderInfoRedis,
    TransactionsRedis,
    BalancesRedis,
    FillsRedis
)
from cryptofeed.backends.arctic import (
    TradeArctic,
    FundingArctic,
    BookArctic,
    TickerArctic,
    OpenInterestArctic,
    LiquidationsArctic,
    CandlesArctic,
    OrderInfoArctic,
    TransactionsArctic,
    BalancesArctic,
    FillsArctic
)
from cryptofeed.backends.elastic import (
    TradeElastic,
    FundingElastic,
    BookElastic,
    TickerElastic,
    OpenInterestElastic,
    LiquidationsElastic,
    CandlesElastic,
    OrderInfoElastic,
    TransactionsElastic,
    BalancesElastic,
    FillsElastic
)
from cryptofeed.backends.clickhouse import (
    TradeClickHouse,
    FundingClickHouse,
    BookClickHouse,
    TickerClickHouse,
    OpenInterestClickHouse,
    LiquidationsClickHouse,
    CandlesClickHouse,
    OrderInfoClickHouse,
    TransactionsClickHouse,
    BalancesClickHouse,
    FillsClickHouse
)
from cryptofeed.backends.postgres import (
    TradePostgres,
    FundingPostgres,
    BookPostgres,
    TickerPostgres,
    OpenInterestPostgres,
    LiquidationsPostgres,
    CandlesPostgres,
    OrderInfoPostgres,
    TransactionsPostgres,
    BalancesPostgres,
    FillsPostgres
)
from cryptofeed.backends.gcp import (
    TradeGCP,
    FundingGCP,
    BookGCP,
    TickerGCP,
    OpenInterestGCP,
    LiquidationsGCP,
    CandlesGCP,
    OrderInfoGCP,
    TransactionsGCP,
    BalancesGCP,
    FillsGCP
)
from cryptofeed.backends.aws import (
    TradeAWS,
    FundingAWS,
    BookAWS,
    TickerAWS,
    OpenInterestAWS,
    LiquidationsAWS,
    CandlesAWS,
    OrderInfoAWS,
    TransactionsAWS,
    BalancesAWS,
    FillsAWS
)
from cryptofeed.backends.azure import (
    TradeAzure,
    FundingAzure,
    BookAzure,
    TickerAzure,
    OpenInterestAzure,
    LiquidationsAzure,
    CandlesAzure,
    OrderInfoAzure,
    TransactionsAzure,
    BalancesAzure,
    FillsAzure
)
from cryptofeed.backends.quest import (
    TradeQuest,
    FundingQuest,
    BookQuest,
    TickerQuest,
    OpenInterestQuest,
    LiquidationsQuest,
    CandlesQuest,
    OrderInfoQuest,
    TransactionsQuest,
    BalancesQuest,
    FillsQuest
)
from cryptofeed.backends.influxdb import (
    TradeInflux,
    FundingInflux,
    BookInflux,
    TickerInflux,
    OpenInterestInflux,
    LiquidationsInflux,
    CandlesInflux,
    OrderInfoInflux,
    TransactionsInflux,
    BalancesInflux,
    FillsInflux
)
from cryptofeed.backends.mongo import (
    TradeMongo,
    FundingMongo,
    BookMongo,
    TickerMongo,
    OpenInterestMongo,
    LiquidationsMongo,
    CandlesMongo,
    OrderInfoMongo,
    TransactionsMongo,
    BalancesMongo,
    FillsMongo
)


BACKEND_MAP = {
    "udp": [UDPCallback],
    "http": [HTTPCallback],
    "socket": [SocketCallback],
    "tcp": [TCPCallback],
    "kafka": [
        TradeKafka,
        FundingKafka,
        BookKafka,
        TickerKafka,
        OpenInterestKafka,
        LiquidationsKafka,
        CandlesKafka,
        OrderInfoKafka,
        TransactionsKafka,
        BalancesKafka,
        FillsKafka
    ],
    "rabbit": [
        TradeRabbit,
        FundingRabbit,
        BookRabbit,
        TickerRabbit,
        OpenInterestRabbit,
        LiquidationsRabbit,
        CandlesRabbit,
        OrderInfoRabbit,
        TransactionsRabbit,
        BalancesRabbit,
        FillsRabbit
    ],
    "redis": [
        TradeRedis,
        FundingRedis,
        BookRedis,
        TickerRedis,
        OpenInterestRedis,
        LiquidationsRedis,
        CandlesRedis,
        OrderInfoRedis,
        TransactionsRedis,
        BalancesRedis,
        FillsRedis
    ],
    "arctic": [
        TradeArctic,
        FundingArctic,
        BookArctic,
        TickerArctic,
        OpenInterestArctic,
        LiquidationsArctic,
        CandlesArctic,
        OrderInfoArctic,
        TransactionsArctic,
        BalancesArctic,
        FillsArctic
    ],
    "elastic": [
        TradeElastic,
        FundingElastic,
        BookElastic,
        TickerElastic,
        OpenInterestElastic,
        LiquidationsElastic,
        CandlesElastic,
        OrderInfoElastic,
        TransactionsElastic,
        BalancesElastic,
        FillsElastic
    ],
    "clickhouse": [
        TradeClickHouse,
        FundingClickHouse,
        BookClickHouse,
        TickerClickHouse,
        OpenInterestClickHouse,
        LiquidationsClickHouse,
        CandlesClickHouse,
        OrderInfoClickHouse,
        TransactionsClickHouse,
        BalancesClickHouse,
        FillsClickHouse
    ],
    "postgres": [
        TradePostgres,
        FundingPostgres,
        BookPostgres,
        TickerPostgres,
        OpenInterestPostgres,
        LiquidationsPostgres,
        CandlesPostgres,
        OrderInfoPostgres,
        TransactionsPostgres,
        BalancesPostgres,
        FillsPostgres
    ],
    "gcp": [
        TradeGCP,
        FundingGCP,
        BookGCP,
        TickerGCP,
        OpenInterestGCP,
        LiquidationsGCP,
        CandlesGCP,
        OrderInfoGCP,
        TransactionsGCP,
        BalancesGCP,
        FillsGCP
    ],
    "aws": [
        TradeAWS,
        FundingAWS,
        BookAWS,
        TickerAWS,
        OpenInterestAWS,
        LiquidationsAWS,
        CandlesAWS,
        OrderInfoAWS,
        TransactionsAWS,
        BalancesAWS,
        FillsAWS
    ],
    "azure": [
        TradeAzure,
        FundingAzure,
        BookAzure,
        TickerAzure,
        OpenInterestAzure,
        LiquidationsAzure,
        CandlesAzure,
        OrderInfoAzure,
        TransactionsAzure,
        BalancesAzure,
        FillsAzure
    ],
    "quest": [
        TradeQuest,
        FundingQuest,
        BookQuest,
        TickerQuest,
        OpenInterestQuest,
        LiquidationsQuest,
        CandlesQuest,
        OrderInfoQuest,
        TransactionsQuest,
        BalancesQuest,
        FillsQuest
    ],
    "influxdb": [
        TradeInflux,
        FundingInflux,
        BookInflux,
        TickerInflux,
        OpenInterestInflux,
        LiquidationsInflux,
        CandlesInflux,
        OrderInfoInflux,
        TransactionsInflux,
        BalancesInflux,
        FillsInflux
    ],
    "mongo": [
        TradeMongo,
        FundingMongo,
        BookMongo,
        TickerMongo,
        OpenInterestMongo,
        LiquidationsMongo,
        CandlesMongo,
        OrderInfoMongo,
        TransactionsMongo,
        BalancesMongo,
        FillsMongo
    ],
    "nats": [
        TradeNATS,
        FundingNATS,
        BookNATS,
        TickerNATS,
        OpenInterestNATS,
        LiquidationsNATS,
        CandlesNATS,
        OrderInfoNATS,
        TransactionsNATS,
        BalancesNATS,
        FillsNATS
    ]
}
