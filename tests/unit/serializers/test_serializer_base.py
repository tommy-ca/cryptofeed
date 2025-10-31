"""
Unit tests for Serializer abstract base class.

Tests verify ABC enforcement, abstract methods, and type safety
according to requirements R2 (Backend Callback Serialization Format Support).
"""
import pytest
from abc import ABC


class TestSerializerABC:
    """Test Serializer abstract base class."""
    
    def test_serializer_cannot_be_instantiated(self):
        """Serializer ABC raises TypeError on direct instantiation."""
        from cryptofeed.serializers import Serializer
        
        with pytest.raises(TypeError, match="abstract"):
            Serializer()
    
    def test_serializer_has_abstract_methods(self):
        """Serializer defines serialize and content_type as abstract."""
        from cryptofeed.serializers import Serializer
        
        # Verify it's an ABC
        assert issubclass(Serializer, ABC)
        
        # Verify abstract methods exist
        assert hasattr(Serializer, 'serialize')
        assert hasattr(Serializer, 'content_type')
        
        # Verify they are abstract
        abstract_methods = Serializer.__abstractmethods__
        assert 'serialize' in abstract_methods
        assert 'content_type' in abstract_methods
    
    def test_incomplete_implementation_fails(self):
        """Incomplete subclass implementation raises TypeError."""
        from cryptofeed.serializers import Serializer
        
        # Only implement serialize, not content_type
        class IncompleteSerializer(Serializer):
            def serialize(self, obj):
                return b'test'
        
        with pytest.raises(TypeError, match="abstract"):
            IncompleteSerializer()
    
    def test_complete_implementation_succeeds(self):
        """Complete subclass implementation instantiates successfully."""
        from cryptofeed.serializers import Serializer
        
        class ConcreteSerializer(Serializer):
            def serialize(self, obj):
                return b'test'
            
            def content_type(self):
                return 'text/plain'
        
        # Should instantiate without error
        serializer = ConcreteSerializer()
        
        # Verify methods work
        assert serializer.serialize("anything") == b'test'
        assert serializer.content_type() == 'text/plain'
    
    def test_serializer_signature(self):
        """Serializer methods have correct signatures."""
        from cryptofeed.serializers import Serializer
        import inspect
        
        # Check serialize signature
        serialize_sig = inspect.signature(Serializer.serialize)
        params = list(serialize_sig.parameters.keys())
        assert 'self' in params
        assert 'data' in params or 'obj' in params
        
        # Check content_type signature
        content_type_sig = inspect.signature(Serializer.content_type)
        params = list(content_type_sig.parameters.keys())
        assert params == ['self']
    
    def test_serializer_return_types(self):
        """Serializer methods have correct return type hints."""
        from cryptofeed.serializers import Serializer
        import inspect
        
        # Get type hints
        serialize_hints = inspect.get_annotations(Serializer.serialize)
        content_type_hints = inspect.get_annotations(Serializer.content_type)
        
        # Verify serialize returns bytes
        if 'return' in serialize_hints:
            assert serialize_hints['return'] == bytes
        
        # Verify content_type returns str
        if 'return' in content_type_hints:
            assert content_type_hints['return'] == str


class TestSerializerSubclassing:
    """Test Serializer subclassing behavior."""
    
    def test_multiple_serializers_coexist(self):
        """Multiple Serializer subclasses can coexist."""
        from cryptofeed.serializers import Serializer
        
        class SerializerA(Serializer):
            def serialize(self, obj):
                return b'A'
            
            def content_type(self):
                return 'type/a'
        
        class SerializerB(Serializer):
            def serialize(self, obj):
                return b'B'
            
            def content_type(self):
                return 'type/b'
        
        a = SerializerA()
        b = SerializerB()
        
        assert a.serialize(None) == b'A'
        assert b.serialize(None) == b'B'
        assert a.content_type() == 'type/a'
        assert b.content_type() == 'type/b'
    
    def test_serializer_isinstance_check(self):
        """Serializer subclasses pass isinstance checks."""
        from cryptofeed.serializers import Serializer
        
        class TestSerializer(Serializer):
            def serialize(self, obj):
                return b'test'
            
            def content_type(self):
                return 'test/type'
        
        serializer = TestSerializer()
        
        assert isinstance(serializer, Serializer)
        assert isinstance(serializer, TestSerializer)
    
    def test_serializer_liskov_substitution(self):
        """Serializer subclasses are substitutable (Liskov Substitution)."""
        from cryptofeed.serializers import Serializer
        
        class SerializerA(Serializer):
            def serialize(self, obj):
                return b'A'
            
            def content_type(self):
                return 'type/a'
        
        class SerializerB(Serializer):
            def serialize(self, obj):
                return b'B'
            
            def content_type(self):
                return 'type/b'
        
        def use_serializer(serializer: Serializer, data):
            """Function that accepts any Serializer."""
            result = serializer.serialize(data)
            mime_type = serializer.content_type()
            return result, mime_type
        
        # Both serializers work with the same function
        result_a = use_serializer(SerializerA(), "test")
        result_b = use_serializer(SerializerB(), "test")
        
        assert result_a == (b'A', 'type/a')
        assert result_b == (b'B', 'type/b')
