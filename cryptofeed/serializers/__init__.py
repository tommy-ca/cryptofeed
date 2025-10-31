'''
Copyright (C) 2017-2025 Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.

Serialization module for cryptofeed backend callbacks.
Provides pluggable serialization formats (JSON, Protobuf, etc.).
'''
from cryptofeed.serializers.base import Serializer
from cryptofeed.serializers.json import JSONSerializer
from cryptofeed.serializers.protobuf import ProtobufSerializer

__all__ = ['Serializer', 'JSONSerializer', 'ProtobufSerializer']
