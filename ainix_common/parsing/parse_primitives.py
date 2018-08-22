from collections import namedtuple
import typecontext
import attr
from typing import List, Callable, Tuple, Dict, Optional


class TypeParser:
    """A TypeParser reads a string and selects the AInixObject
    implementation that the string represents. It also selects the substring
    that will be fed into the ObjectParser for the chosen implementation.

    Args:
          type_context : context this parser exists in.
          parser_name : the string identifier to use for this parser
          type_name : string identifier of the type this parser parses.
          parse_function : a callable that is used when parsing the string.
            when self.parse_string is called this function will be called
            with three arguments provided.
                1) This parser (self)
                2) The string we would like to parse
                3) An instance of TypeParserResult where the callable should put
                   its result.
    """
    def __init__(
        self,
        type_context: 'typecontext.TypeContext',
        parser_name: str,
        type_name: str,
        parse_function: Callable[['TypeParser', str, 'TypeParserResult'], None]
    ):
        self._type_context = type_context
        if not parser_name:
            raise ValueError("TypeParser parser_name should be "
                             "non-None and non-empty")
        self.type_name = type_name
        self.parser_name = parser_name
        self._parse_function = parse_function
        self._type = None
        self._type_implementations = None
        self._type_context.register_type_parser(self)

    @staticmethod
    def match_attribute(
        object_list: List['typecontext.AInixObject'],
        key: str,
        value: str
    ) -> List['typecontext.AInixObject']:
        """Helper that filters a list objects that only have a certain attribute

        Args:
            object_list : list to filter.
            key : the string key of the attribute to filter on.
            value : the value we want to key to match on.
        """
        return [o for o in object_list if o.type_data[key] == value]

    def _resolve_type(self) -> None:
        """Sets the internal reference to the actual python object_name
        reference to the type from the type's string name,"""
        if self._type is None:
            self._type = self._type_context.get_type_by_name(self.type_name)

    @property
    def type_implementations(self) -> List['typecontext.AInixObject']:
        """Returns list of AInix objects that implement this parser's type"""
        self._resolve_type()
        if self._type_implementations is None:
            self._type_implementations = \
                self._type_context.get_implementations(self.type_name)
        return self._type_implementations

    def parse_string(self, string: str) -> 'TypeParserResult':
        """
        Args:
            string : a string that you would like would like to parse and get
                the implementation of.
        """
        self._resolve_type()
        result = TypeParserResult(self._type, string)
        self._parse_function(self, string, result)
        return result


class TypeParserResult:
    """Stores the result of TypeParser.parse_string().

    Args:
        type_instance : type of the result.
        string : string we parsed.
    """
    def __init__(self, type_instance: 'typecontext.AInixType', string: str):
        self.type = type_instance
        self.string = string
        self._implementation: typecontext.AInixObject = None
        self._next_slice = None
        self._next_parser = None

    def get_implementation(self) -> 'typecontext.AInixObject':
        return self._implementation

    def get_next_string(self):
        si, ei = self._next_slice
        return self.string[si:ei].strip()

    def set_valid_implementation(self, implementation):
        self._implementation = implementation

    def set_next_slice(self, start_idx, end_idx):
        self._next_slice = (int(start_idx), int(end_idx))

    @property
    def next_parser(self) -> 'ObjectParser':
        """Gets the ObjectParser for the parsed object_name. Note, this
        should only be called after a call to set_valid_implementation
        and is primarily intended for receivers of the result, not
        for the parser itself
        """
        if self._implementation is None:
            raise AInixParseError("No implementation set during parsing")
        next_parser = self._implementation.preferred_object_parser
        if next_parser is None:
            next_parser = self.type.default_object_parser
        # TODO (DNGros): allow parsers to override the object parser
        return next_parser


class ObjectParser:
    """An ObjectParser reads a string and determines which of the object's
    arguments are present. If the argument is present, it also selects the
    substring that will be fed into the TypeParser for the argument (assuming
    the argument has a non-None type).

    Args:
          type_context : context this parser exists in.
          parser_name : string identifier for this parser
          type_name : the string identifier representing the
          parse_function : a callable that is used when parsing the string.
            when self.parse_string() is called this function will be called
            with three arguments provided.
                1) This parser (self)
                2) The object instance we are parsing
                3) The string we would like to parse
                4) An instance of ObjectParserResult which the callable should
                   interact with to store the result of the parse
            The callable itself is not expected to return any value.
    """
    def __init__(
        self,
        type_context: 'typecontext.TypeContext',
        parser_name: str,
        type_name: str,
        parse_function: Callable[['ObjectParser', 'typecontext.AInixObject',
                                  str, 'ObjectParserResult'], None]
    ):
        self.name = parser_name
        self.parser_name = parser_name
        self._type_context = type_context
        self.type_name = type_name
        self._parse_function = parse_function
        self._type = None
        self._type_context.register_object_parser(self)

    def _resolve_type(self) -> None:
        """Sets the internal reference to the actual python object_name
        reference to the type from the type's string name,"""
        if self._type is None:
            self._type = self._type_context.get_type_by_name(self.type_name)

    def parse_string(
        self,
        object_: 'typecontext.AInixObject',
        string: str
    ) -> 'ObjectParserResult':
        if object_.type_name != self.type_name:
            raise ValueError("ObjectParser {0.name} expects to parse objects"
                             "of type {0.type_name}, but parse_string called"
                             "with object of type {0.type_name}".format(self))
        self._resolve_type()
        result = ObjectParserResult(object_, string)
        self._parse_function(self, object_, string, result)
        return result


@attr.s(auto_attribs=True)
class ObjectParseArgData:
    slice: Tuple[int, int]
    slice_string: str

class ObjectParserResult:
    def __init__(self, object_to_parse: 'typecontext.AInixObject', string: str):
        self._object = object_to_parse
        self._result_dict: Dict[str, ObjectParseArgData] = \
            {arg.name: None for arg in object_to_parse.children}
        self._sibling_result = None
        self.string = string

    def _get_slice_string(self, start_idx: int, end_idx: int) -> str:
        return self.string[start_idx:end_idx].strip()

    def get_arg_present(self, name) -> Optional[ObjectParseArgData]:
        return self._result_dict.get(name, None)

    def set_arg_present(self, arg_name: str, start_idx: int, end_idx: int):
        # TODO (DNGros): check that the argname exists
        si, ei = int(start_idx), int(end_idx)
        if arg_name not in self._result_dict:
            raise AInixParseError(f"Invalid argument name {arg_name} for parse "
                                  f"result of {self._object.name}. Valid options"
                                  f"are [{', '.join(self._result_dict.keys())}]")
        self._result_dict[arg_name] = ObjectParseArgData(
            (si, ei), self._get_slice_string(si, ei))


class AInixParseError(RuntimeError):
    pass


def SingleTypeImplParserFunc(
    parser: TypeParser,
    string: str,
    result: TypeParserResult
) -> None:
    """The parser function for a special builtin TypeParser which works when
    there is only one implementation of the specified type"""
    num_impls = len(parser.type_implementations)
    if num_impls != 1:
        raise AInixParseError(f"{parser.parser_name} expects exactly one "
                              f"implementation. Found {num_impls} implementations.")
    result.set_valid_implementation(parser.type_implementations[0])
    result.set_next_slice(0, len(string))
