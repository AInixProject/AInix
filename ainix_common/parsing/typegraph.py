import parse_primitives
print(parse_primitives.__dict__)
from parse_primitives import ObjectParser, TypeParser
from collections import defaultdict
from typing import List, Type


class TypeGraph:
    class AInixType:
        """Used to construct AInix types.

        Types represent a collection of related objects.

        Args:
            name : An upper CamelCase string to identify this type
        """
        def __init__(
            self,
            type_graph: 'TypeGraph',
            name: str,
            default_type_parser: Type[TypeParser],
            default_object_parser: Type[ObjectParser],
            allowed_attributes
        ):
            self.type_graph = type_graph
            self.name = name
            self.default_type_parser = default_type_parser
            self.default_object_parser = default_object_parser
            self.allowed_attributes = allowed_attributes

        def __eq__(self, other):
            return other.name == self.name and other.type_graph is self.type_graph

    class AInixObject:
        def __init__(
            self,
            name: str,
            type: 'AInixType',
            children: List['AInixArgument'],
            direct_sibling: 'AInixArgument',
            type_data: dict
        ):
            self.name = name
            self.type = type
            self.children = children
            self.type_data = type_data
            self.direct_sibling = direct_sibling

    def __init__(self):
        self.name_to_type = {}
        self.name_to_object = {}
        self.type_name_to_implementations = defaultdict(list)

    def _resolve_type(self, type):
        """Converts a string into a type object if needed"""
        if isinstance(type, str):
            got_type = self.get_type_by_name(type)
            if got_type is None:
                raise ValueError("Unable to find type", type)
            return got_type
        return type

    def get_type_by_name(self, name):
        return self.name_to_type.get(name, None)

    def get_object_by_name(self, name):
        return self.name_to_object.get(name, None)

    def get_implementations(self, type):
        type = self._resolve_type(type)
        return self.type_name_to_implementations[type.name]

    def create_type(
        self,
        name: str,
        default_type_parser: TypeParser = None,
        default_object_parser:ObjectParser = None,
        allowed_attributes = []
    ) -> 'AInixType':
        if name in self.name_to_type:
            raise ValueError("Type", name, "already exists")
        new_type = self.AInixType(self, name, default_type_parser,
                                  default_object_parser, allowed_attributes)
        self.name_to_type[name] = new_type
        return new_type

    def create_object(
        self,
        name: str,
        type,
        children: list,
        direct_sibling: 'AInixArgument' = None,
        type_data: dict = None
    ) -> AInixObject:
        if name in self.name_to_object:
            raise ValueError("Object", name, "already exists")
        type = self._resolve_type(type)
        if type_data is None:
            type_data = {}
        new_object = self.AInixObject(name, type, children, direct_sibling, type_data)
        self.name_to_object[name] = new_object
        self.type_name_to_implementations[type.name].append(new_object)
        return new_object


class AInixArgument:
    def __init__(
        self,
        name: str,
        type: TypeGraph.AInixType,
        type_parser: Type[TypeParser] = None,
        required: bool = False,
        arg_data: dict = {}
    ):
        self.name = name
        self.type = type
        self.required = required
        if type is not None:
            if type_parser is not None:
                self.type_parser = type_parser(type)
            else:
                if type.default_type_parser is None:
                    raise ValueError("No type_parser provided for an AInixArgument. \
                        However, type {self.type.name} does not provide a default \
                        type_parser")
                self.type_parser = type.default_type_parser(type)
        self.arg_data = arg_data
