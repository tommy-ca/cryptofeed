'''
Copyright (C) 2017-2025 Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.

Tests for Serializer Abstract Base Class
'''
import pytest
from cryptofeed.serializers.base import Serializer


def test_serializer_cannot_be_instantiated():
    """Serializer ABC raises TypeError on direct instantiation."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        Serializer()


def test_serializer_has_abstract_methods():
    """Serializer defines serialize and content_type as abstract."""
    assert hasattr(Serializer, 'serialize')
    assert hasattr(Serializer, 'content_type')
    assert Serializer.serialize.__isabstractmethod__
    assert Serializer.content_type.__isabstractmethod__


def test_incomplete_implementation_fails():
    """Incomplete subclass implementation raises TypeError."""
    class IncompleteSerializer(Serializer):
        def serialize(self, obj):
            return b'test'
        # Missing content_type()

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        IncompleteSerializer()


def test_complete_implementation_succeeds():
    """Complete subclass implementation instantiates successfully."""
    class ConcreteSerializer(Serializer):
        def serialize(self, obj):
            return b'test'
        
        def content_type(self):
            return 'text/plain'

    s = ConcreteSerializer()
    assert s.serialize("anything") == b'test'
    assert s.content_type() == 'text/plain'


def test_serializer_type_hints():
    """Serializer methods have proper type hints."""
    import inspect
    
    # Check serialize signature
    sig = inspect.signature(Serializer.serialize)
    assert 'obj' in sig.parameters
    assert sig.return_annotation == bytes
    
    # Check content_type signature
    sig = inspect.signature(Serializer.content_type)
    assert len(sig.parameters) == 1  # Only self
    assert sig.return_annotation == str
