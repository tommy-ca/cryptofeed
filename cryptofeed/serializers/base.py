"""
Abstract base class for serializers.

Defines the contract that all serializers must implement, following the
Open/Closed and Liskov Substitution principles.
"""
from abc import ABC, abstractmethod
from typing import Any


class Serializer(ABC):
    """Abstract base class for data serialization.
    
    Subclasses must implement serialize() and content_type() methods.
    This design follows SOLID principles:
    - Single Responsibility: Only handles serialization contract
    - Open/Closed: Open for extension (new formats), closed for modification
    - Liskov Substitution: All subclasses are substitutable
    - Interface Segregation: Minimal interface (2 methods)
    - Dependency Inversion: Code depends on abstraction, not concrete classes
    
    Example:
        class JSONSerializer(Serializer):
            def serialize(self, data: Any) -> bytes:
                return json.dumps(data).encode('utf-8')
            
            def content_type(self) -> str:
                return 'application/json'
    """
    
    @abstractmethod
    def serialize(self, data: Any) -> bytes:
        """Convert data object to serialized bytes.
        
        Args:
            data: Data object to serialize (e.g., Trade, OrderBook, dict)
        
        Returns:
            bytes: Serialized data in the format implemented by this serializer
        
        Raises:
            SerializationError: If serialization fails
        """
        pass
    
    @abstractmethod
    def content_type(self) -> str:
        """Return MIME type for this serialization format.
        
        Returns:
            str: MIME type string (e.g., 'application/json', 'application/x-protobuf')
        """
        pass
