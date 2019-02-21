"""This module contains the basic classes for parsing. It defines the different
kinds of parsers, as well as various helping classes for managing the data
coming from parsers."""
from ainix_common.parsing import typecontext
import attr
from typing import List, Callable, Tuple, Dict, Optional, Any, Union, Generator
import typing
import types

TypeParserFuncType = Callable[
    ['TypeParserRun', str, 'TypeParserResult'],
    Union[Generator['ImplementationParseDelegation', 'ParseDelegationReturnMetadata', None], None]
]
TypeParserToStringFuncType = Callable[['TypeToStringResult'], Generator]


def DefaultTypeToString(
    result: 'TypeToStringResult'
):
    result.add_impl_unparse()
    

class TypeParser:
    """A TypeParser reads a string and selects the AInixObject
    implementation that the string represents. It also selects the substring
    that will be fed into the ObjectParser for the chosen implementation.

    Args:
          type_context : context this parser exists in.
          parser_name : the string identifier to use for this parser
          parse_function : a callable that is used when parsing the string.
            when self.parse_string is called this function will be called
            with three arguments provided.
                1) A instance of a TypeParserRun. This provides utility functions
                   to the parser function to use while parsing.
                2) The string we would like to parse
                3) An instance of TypeParserResult where the callable should put
                   its result.
          to_string_function: A callable which is used for unparsing (going from
            a parsed AST back to a string).
          type_name : string identifier of the type this parser parses. If None
            is given, then this parser is able to work used with any type. The
            type is provided we are parsing is passed in with calls to parse_string
    """
    def __init__(
        self,
        type_context: 'typecontext.TypeContext',
        parser_name: str,
        parse_function: TypeParserFuncType,
        to_string_function: TypeParserToStringFuncType = DefaultTypeToString,
        type_name: str = None
    ):
        self._type_context = type_context
        if not parser_name:
            raise ValueError("TypeParser parser_name should be "
                             "non-None and non-empty")
        if to_string_function is None:
            to_string_function = DefaultTypeToString
        self.type_name = type_name
        self.parser_name = parser_name
        self._parse_function = parse_function
        self._type = None
        self._type_implementations = None
        self._type_context.register_type_parser(self)
        self._to_string_func = to_string_function

    def _resolve_type(self) -> None:
        """Sets the internal reference to the actual python object_name
        reference to the type from the type's string name,"""
        if self._type is None and self.type_name is not None:
            self._type = self._type_context.get_type_by_name(self.type_name)

    def parse_string(
        self,
        string: str,
        type_to_parse: 'typecontext.AInixType' = None
    ) -> typing.Generator[
        'ImplementationParseDelegation',
        'ParseDelegationReturnMetadata',
        'TypeParserResult'
    ]:
        """
        Args:
            string : a string that you would like would like to parse and get
                the implementation of.
            type_to_parse : Instance of the type we are parsing. This is
                only required if the TypeParser was constructed without a
                specific type.
        """
        self._resolve_type()
        result_type = self._type
        if self._type is None:
            if type_to_parse is None:
                raise AInixParseError(f"{self} was not constructed with"
                                      f" a specific type specified. You must "
                                      f"pass in what type you are parsing.")
            result_type = type_to_parse
        elif type_to_parse is not None:
            if self.type_name != type_to_parse.name:
                raise AInixParseError(f"{self} expects to parse type "
                                      f"{self.type_name} but parser_string() "
                                      f"given {type_to_parse}")
        run = TypeParserRun(self._type_context, result_type, self, string)
        result = TypeParserResult(result_type, string)
        parse_func_call = self._parse_function(run, string, result)
        if isinstance(parse_func_call, types.GeneratorType):
            yield from parse_func_call
        self._validate_parse_result(result)
        return result

    def to_string(
        self,
        implementation_chosen: 'typecontext.AInixObject',
        type_of_chosen: 'typecontext.AInixType'
    ) -> 'TypeToStringResult':
        result = TypeToStringResult(implementation_chosen, type_of_chosen)
        self._to_string_func(result)
        return result

    def _validate_parse_result(self, result: 'TypeParserResult'):
        """Check if the result we have after running the parse func is valid"""
        is_valid = result.get_implementation() is not None
        if not is_valid:
            raise AInixParseError(f"{self} did not set a valid implementation.")

    def __str__(self):
        return f"<TypeParser {self.parser_name}>"


class TypeParserRun:
    """Provides a collection of utility functions that is provided to a parse
    function to help with the parsing."""
    def __init__(
        self,
        type_context: 'typecontext.TypeContext',
        type_instance: 'typecontext.AInixType',
        parser: TypeParser,
        string_to_parse: str
    ):
        self._type = type_instance
        self._type_context = type_context
        self.parser_name = parser.parser_name
        self._type_implementations = None
        self._string = string_to_parse

    @staticmethod
    def match_attribute(
        object_list: List['typecontext.AInixObject'],
        key: str,
        value: str
    ) -> List['typecontext.AInixObject']:
        """Helper that filters a list objects that only have a certain type_data
        attribute

        Args:
            object_list : list to filter.
            key : the string key of the attribute to filter on.
            value : the value we want to key to match on.
        """
        return [o for o in object_list if o.type_data[key] == value]

    @property
    def all_type_implementations(self) -> List['typecontext.AInixObject']:
        """Returns list of AInix objects that could be in this runs results result"""
        if self._type_implementations is None:
            self._type_implementations = \
                self._type_context.get_implementations(self._type.name)
        return self._type_implementations

    def delegate_parse_implementation(
        self,
        implementation: 'typecontext.AInixObject',
        slice_to_parse: Tuple[int, int]
    ) -> 'ImplementationParseDelegation':
        """Create delegation to attempt to parse some implementation of the type.
        This will return an object which should then be yielded. The StringParser"""
        si, endi = slice_to_parse
        return ImplementationParseDelegation(
            implementation=implementation,
            string_to_parse=self._string[si: endi],
            slice_to_parse=slice_to_parse,
            next_parser=_next_parser_of_impl(implementation, self._type)
        )


class TypeParserResult:
    """Stores the result of TypeParser.parse_string(). It also contains
    several convience functions to help a parser function while parseing.

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
        self._accepted_delegation = None

    def get_implementation(self) -> 'typecontext.AInixObject':
        return self._implementation

    def get_next_string(self):
        si, ei = self._next_slice
        return self.string[si:ei]

    def get_next_slice(self) -> Tuple[int, int]:
        """Gets the slice of the input string used to parse the implementation
        that we chose"""
        return self._next_slice

    def set_valid_implementation(self, implementation: 'typecontext.AInixObject'):
        if not isinstance(implementation, typecontext.AInixObject):
            raise AInixParseError(f"Expected a object. Got a {implementation}")
        if implementation.type != self.type:
            raise AInixParseError("Wrong type! How did that happen?")
        self._implementation = implementation

    def set_valid_implementation_name(self, impl_name: str):
        self._implementation = self.type.type_context.get_object_by_name(impl_name)

    def set_next_slice(self, start_idx, end_idx, strip_slice: bool = False):
        """Sets the slice of the input string used to parse the implementation
        that we chose"""
        si, ei = int(start_idx), int(end_idx)
        if strip_slice:
            si, ei = _strip_slice_of_string(self.string, si, ei)
        self._next_slice = (si, ei)

    @property
    def next_parser(self) -> 'ObjectParser':
        """Gets the ObjectParser for the parsed object_name. Note, this
        should only be called after a call to set_valid_implementation
        and is primarily intended for receivers of the result, not
        for the parser itself
        """
        # TODO (DNGros): allow parsers to override the object parser or keep
        # track of it at the parser level
        return _next_parser_of_impl(self._implementation, self.type)

    def accept_delegation(self, delegation_resp: 'ParseDelegationReturnMetadata'):
        if not delegation_resp.parse_success:
            raise ValueError("Tried to accept unsucessful delegation")
        self._implementation = delegation_resp.what_parsed
        self._accepted_delegation = delegation_resp
        si = delegation_resp.remaining_right_starti + delegation_resp.original_start_offset
        endi = len(self.string)
        self.set_next_slice(si, endi)


class TypeToStringResult:
    def __init__(
        self,
        implementation: 'typecontext.AInixObject',
        type_of_impl: 'typecontext.AInixType'
    ):
        self._unparse_seq: List[Union[str, ImplementationToStringDelegation]] = []
        self._implementation = implementation
        self._type_of_impl = type_of_impl
        self._already_added_impl = False

    @property
    def implementation(self):
        return self._implementation

    @property
    def unparse_seq(self) -> Tuple[Union[str, 'ImplementationToStringDelegation'],]:
        return tuple(self._unparse_seq)

    @property
    def type_of_impl(self):
        return self._type_of_impl

    def add_string(self, string: str):
        self._unparse_seq.append(string)

    def add_impl_unparse(self):
        if self._already_added_impl:
            raise ValueError("Cannot add implementation multiple times in unparse")
        next_parser = _next_parser_of_impl(self._implementation, self._type_of_impl)
        to_string_del = ImplementationToStringDelegation(next_parser)
        self._unparse_seq.append(to_string_del)
        self._already_added_impl = True


def _next_parser_of_impl(
    implementation: 'typecontext.AInixObject',
    type_: 'typecontext.AInixType'
):
    if implementation is None:
        raise ValueError("No impl provided")
    next_parser = implementation.preferred_object_parser
    if next_parser is None:
        next_parser = type_.default_object_parser
    return next_parser


@attr.s(auto_attribs=True)
class ImplementationParseDelegation:
    """Represents a delegation back to the calling parser asking to parse
    an implementation object and see if it succeeds."""
    implementation: 'typecontext.AInixObject'
    string_to_parse: str
    slice_to_parse: Tuple[int, int]
    next_parser: 'ObjectParser'


@attr.s(auto_attribs=True, frozen=True)
class ImplementationToStringDelegation:
    """Used in an unparse to represent the slot where the implentation goes"""
    next_parser: 'ObjectParser'


# Monstrosity that describes the type of a parser_func for Object parsers.
# See the docstring of ObjectParser for more info.
ObjectParseFuncType = Callable[['ObjectParserRun', str, 'ObjectParserResult'],
    Union[typing.Generator['ArgParseDelegation', 'ParseDelegationReturnMetadata', None], None]]

ObjectToStringFuncType = Callable[['ObjectNodeArgMap', 'ObjectToStringResult'], None]


class ObjectParser:
    """An ObjectParser reads a string and determines which of the object's
    arguments are present. If the argument is present, it also selects the
    substring that will be fed into the TypeParser for the argument (assuming
    the argument has a non-None type).

    Args:
          type_context : context this parser exists in.
          parser_name : string identifier for this parser
          parse_function : a callable that is used when parsing the string.
            when self.parse_string() is called this function will be called
            with three arguments provided.
                1) An instance of ObjectParseRun which provides helpful functions
                   in the context of this call.
                2) The string we would like to parse
                3) An instance of ObjectParserResult which the callable should
                   interact with to store the result of the parse
            The callable itself is not expected to return any value. However,
            it may optionally be a generator which "delegates" the parsing of
            arguments. This can be used to do a "LL" style parsing.
          exclusive_type_name : An optional string. If supplied, this object
            parser will only work on objects of that type.
    """
    def __init__(
        self,
        type_context: 'typecontext.TypeContext',
        parser_name: str,
        parse_function: ObjectParseFuncType,
        to_string_function: ObjectToStringFuncType,
        exclusive_type_name: str = None
    ):
        self.name = parser_name
        self.parser_name = parser_name
        self._type_context = type_context
        self.type_name = exclusive_type_name
        self._parse_function = parse_function
        self._type_context.register_object_parser(self)
        self._to_string_func = to_string_function

    def parse_string(
        self,
        string: str,
        object_: 'typecontext.AInixObject'
    ) -> typing.Generator[
        'ArgParseDelegation', 'ParseDelegationReturnMetadata', 'ObjectParserResult'
    ]:
        if self.type_name is not None and object_.type_name != self.type_name:
            raise ValueError("ObjectParser {0.name} expects to parse objects"
                             "of type {0.type_name}, but parse_string called "
                             "with object of type {0.type_name}".format(self))
        run = ObjectParserRun(object_, string)
        result = ObjectParserResult(object_, string)
        parse_func_run = self._parse_function(run, string, result)
        # Parsers are able to yield delegating parsing things to other parsers.
        # If that is the case, it will be generator.
        if isinstance(parse_func_run, types.GeneratorType):
            yield from parse_func_run

        self._validate_parse_result(result, object_)
        return result

    def to_string(self, arg_map: 'ObjectNodeArgMap') -> 'ObjectToStringResult':
        result = ObjectToStringResult(arg_map)
        self._to_string_func(arg_map, result)
        return result

    def _validate_parse_result(
        self,
        result: 'ObjectParserResult',
        object_parsed: 'typecontext.AInixObject'
    ) -> None:
        """Looks at a ObjectParserResult and verifies that the parse_func returned
        a valid result"""
        for arg in object_parsed.children:
            if arg.required and not result.get_arg_present(arg.name):
                raise AInixParseError(f"In object parser {self.name} arg "
                                      f"{arg.name} was not set but it is a "
                                      f"required arg")


@attr.s(auto_attribs=True, frozen=True)
class ArgParseDelegation:
    """A struct to hold data about requesting that the parsing of an argument
    be handled by some other parser while getting the results reported back.
    This is constructed using methods in the ObjectParserRun

    Args:
        arg: the argument we want parsed
        object_on: The object the argument is of
        string_to_parse: The actual string we are parsing.
        slice_to_parse: The slice into the origional string of the objectparser
            which we would like to parser.
    """
    arg: 'typecontext.AInixArgument'
    object_on: 'typecontext.AInixObject'
    string_to_parse: str
    slice_to_parse: Tuple[int, int]

    def next_from_substring(self, new_string) -> 'ParseDelegationReturnMetadata':
        """Makes a return from a substring. Probably only useful in test"""
        return ParseDelegationReturnMetadata.create_from_substring(
            self.arg, self.string_to_parse, new_string, self.slice_to_parse[0])


@attr.s(auto_attribs=True, frozen=True)
class ParseDelegationReturnMetadata:
    """Used to pass around metadata about the a delegation of parsing to
    another parser.

    Args:
        parse_success: whether or not the parsing succeeded
        string_parsed: The origional string we were parsing
        original_start_offset: The offset into the outer scope string that we
            are currently parsing.
        what_parsed: The instance of the thing we parsed
        remaining_right_starti: The index into the string_parsed which points
            to the remaining unconsumed part of the string_parsed
        fail_reason: If the parse was not successful, this can be used to provide
            a reason for the failure
    """
    parse_success: bool
    string_parsed: str
    original_start_offset: Optional[int]
    what_parsed: Any
    remaining_right_starti: Optional[int]
    fail_reason: str = ""

    @classmethod
    def create_from_substring(
        cls,
        what_parsed,
        original_string: str,
        new_string: str,
        start_offset: int
    ):
        """Constructs a successful return value while calculating for you the
        correct remaining_right_starti based off a new remaining string. Useful
        in test to quickly be able to make expected values"""
        if new_string is None:
            return ParseDelegationReturnMetadata.make_failing()
        if new_string != "":
            remaining_right_starti = original_string.rfind(new_string)
        else:
            remaining_right_starti = len(original_string)
        if remaining_right_starti == -1:
            raise ValueError("New string must be substring of original string")
        if original_string[remaining_right_starti:] != new_string:
            raise ValueError(f"new string must appear at the end of the origional string."
                             f"original string is '{original_string}' new string is '{new_string}'"
                             f"remaining_right_starti is {remaining_right_starti}")
        return cls(True, original_string, start_offset, what_parsed, remaining_right_starti)

    @property
    def remaining_string(self) -> Optional[str]:
        if self.parse_success:
            return self.string_parsed[self.remaining_right_starti:]
        else:
            return None

    @staticmethod
    def make_for_unparsed_string(
        string: str,
        what_parsed,
        original_start_offset: int = 0
    ) -> 'ParseDelegationReturnMetadata':
        """A return value with nothing consumed yet

        Args:
            string: The unparsed string
            what_parsed: What we are parsing. Should be an Arg, Object, or Type
            original_start_offset: The offset into the original object this return happened on
        """
        return ParseDelegationReturnMetadata(True, string, original_start_offset, what_parsed, 0)

    @staticmethod
    def make_failing(
        parsed_string: str = None,
        fail_reason: str = ""
    ) -> 'ParseDelegationReturnMetadata':
        return ParseDelegationReturnMetadata(False, parsed_string, None, None, None, fail_reason)

    def combine_with_child_return(
        self,
        child_return_data: 'ParseDelegationReturnMetadata',
        child_start_offset: int
    ) -> 'ParseDelegationReturnMetadata':
        """When being used during parsing of an ObjectParser, this can be used
        to combine an exisiting return with the return of one of the children."""
        is_success = self.parse_success and child_return_data.parse_success
        fail_message = self.fail_reason + child_return_data.fail_reason
        new_remaining_starti = max(self.remaining_right_starti,
                                   child_return_data.remaining_right_starti + child_start_offset)
        new_start_offset = min(self.original_start_offset, child_return_data.original_start_offset)
        return ParseDelegationReturnMetadata(
            is_success, self.string_parsed, new_start_offset, self.what_parsed,
            new_remaining_starti, fail_message)

    def change_what_parsed(self, new_what_parsed) -> 'ParseDelegationReturnMetadata':
        """Returns a new return with same values accept different what_parsed"""
        return ParseDelegationReturnMetadata(self.parse_success, self.string_parsed,
                                             self.original_start_offset,
                                             new_what_parsed, self.remaining_right_starti,
                                             self.fail_reason)

    def add_fail(self, new_fail_reason) -> 'ParseDelegationReturnMetadata':
        """Returns with a new instance with an additiona fail message added"""
        new_fail_string = " >>> " + self.fail_reason + "\n >>> " + new_fail_reason
        return ParseDelegationReturnMetadata(
            False, self.string_parsed, 0, self.what_parsed, None, new_fail_string)


class ObjectParserRun:
    """Represents one call to the parse function of a ObjectParser. Contains
    utility functions to use while parsing."""
    def __init__(self, object_parsing: 'typecontext.AInixObject', string_to_parse: str):
        self._object = object_parsing
        self._string_to_parse = string_to_parse
        self.all_arguments = self._object.children

    def get_arg_by_name(self, name: str):
        return self._object.get_arg_by_name(name)

    def left_fill_arg(
        self,
        arg: 'typecontext.AInixArgument',
        slice_to_parse: Tuple[int, int]
    ) -> ArgParseDelegation:
        start, end = slice_to_parse
        if arg is None:
            raise ValueError("Can't delegate an arg that is None!")
        return ArgParseDelegation(arg, self._object,
                                  self._string_to_parse[start:end], slice_to_parse)


@attr.s(auto_attribs=True, frozen=True)
class ObjectParseArgData:
    slice: Tuple[int, int]
    slice_string: str
    set_from_delegation: ParseDelegationReturnMetadata = None


def _strip_slice_of_string(base_string, start_i, end_i) -> Tuple[int, int]:
    """Given a slice of a string, returns a new slice into that string with
    left and right whitespace stripped out"""
    while start_i < len(base_string) and base_string[start_i] == " ":
        start_i += 1
    while end_i > start_i and base_string[end_i - 1] == " ":
        end_i -= 1
    return start_i, end_i


class ObjectParserResult:
    def __init__(self, object_to_parse: 'typecontext.AInixObject', string: str):
        self._object = object_to_parse
        self._result_dict: Dict[str, ObjectParseArgData] = \
            {arg.name: None for arg in object_to_parse.children}
        self._sibling_result = None
        self.string = string
        self._remaining_start_i = 0

    def _get_slice_string(self, start_idx: int, end_idx: int) -> str:
        return self.string[start_idx:end_idx]

    def get_arg_present(self, name) -> Optional[ObjectParseArgData]:
        if name not in self._result_dict and self._object.get_arg_by_name("name") is None:
            raise ValueError(f"Arg name {name} not present in {self._object}")
        return self._result_dict.get(name, None)

    def set_arg_present(
        self,
        arg_name: str,
        start_idx: int,
        end_idx: int,
        strip_slice: bool = False
    ):
        si, ei = int(start_idx), int(end_idx)
        if strip_slice:
            si, ei = _strip_slice_of_string(self.string, si, ei)
        if arg_name not in self._result_dict:
            raise AInixParseError(f"Invalid argument name {arg_name} for parse "
                                  f"result of {self._object.name}. Valid options"
                                  f"are [{', '.join(self._result_dict.keys())}]")
        self._result_dict[arg_name] = ObjectParseArgData(
            (si, ei), self._get_slice_string(si, ei))
        self.remaining_start_i = max(self.remaining_start_i, ei)

    @property
    def remaining_start_i(self):
        return self._remaining_start_i

    @remaining_start_i.setter
    def remaining_start_i(self, new_remaining_i: int):
        if new_remaining_i < self.remaining_start_i:
            raise ValueError("Don't really expect you set remaining to be less than already"
                             "is. Idk.. maybe this fine...")
        self._remaining_start_i = new_remaining_i

    def accept_delegation(self, delegation_return: ParseDelegationReturnMetadata):
        if not isinstance(delegation_return.what_parsed, typecontext.AInixArgument):
            raise ValueError(f"Unable to accept a delegation that parsed a "
                             f"{delegation_return.what_parsed}. Must be an AInixArgument")
        if not delegation_return.parse_success:
            raise ValueError("Can't accept a failed delegation.\n"
                             "Fail reason " + delegation_return.fail_reason)
        # TODO (DNGros): slice is probably wrong
        start_i = delegation_return.original_start_offset
        end_i = start_i + delegation_return.remaining_right_starti
        self._result_dict[delegation_return.what_parsed.name] = ObjectParseArgData(
            slice=(start_i, end_i),
            slice_string=self.string[start_i: end_i],
            set_from_delegation=delegation_return
        )
        self.remaining_start_i = max(self.remaining_start_i, end_i)


class ObjectNodeArgMap:
    """A simple container which tracks which arguments are present of an object.
    Is intended for use in unparsing."""
    def __init__(
        self,
        implenetation: 'typecontext.AInixObject',
        is_present_map: Dict[str, bool]
    ):
        self.implementation = implenetation
        self._is_present_map: Dict[str, bool] = is_present_map

    def is_argname_present(self, arg_name: str) -> bool:
        return self._is_present_map[arg_name]


class ObjectToStringResult:
    def __init__(self, arg_map: ObjectNodeArgMap):
        self._arg_map = arg_map
        self.unparse_seq = []

    def add_string(self, string: str):
        self.unparse_seq.append(string)

    def add_arg_tostring(self, arg: 'typecontext.AInixArgument'):
        self.unparse_seq.append(ArgToStringDelegation(arg))

    def add_arg_present_string(self, arg: 'typecontext.AInixArgument', string: str):
        self.unparse_seq.append(ArgIsPresentToString(arg, string))

    def add_argname_tostring(self, arg_name: str):
        self.add_arg_tostring(self._arg_map.implementation.get_arg_by_name(arg_name))

    def make_snapshot(self) -> 'ObjectToStringSnapshot':
        return ObjectToStringSnapshot(self)

class ObjectToStringSnapshot:
    """A helper for saving the state of a result, and jumping back to the checkpoint
    if needed."""
    def __init__(self, result_it_is_of: ObjectToStringResult):
        self._unparse_seq = [x for x in result_it_is_of.unparse_seq]
        self._result = result_it_is_of

    def restore(self):
        """Unrolls the result back to this snapshot"""
        self._result.unparse_seq = self._unparse_seq


@attr.s(auto_attribs=True, frozen=True)
class ArgToStringDelegation:
    """Used in a ObjectToStringResult to indicate where an arg value goes"""
    arg: 'typecontext.AInixArgument'


@attr.s(auto_attribs=True, frozen=True)
class ArgIsPresentToString:
    """Represents extra string data that a ObjectTostring methods can through
    in which represents strings that was used to decide whether an arg is
    present or not."""
    arg: 'typecontext.AInixArgument'
    string: str


class AInixParseError(RuntimeError):
    """An exception for when something goes wrong during parsing. This is
    potentially recoverable if one of the parent nodes that delegated the
    parse expects an error and is able to catch it."""
    pass


class StringProblemParseError(AInixParseError):
    """Exception for when a something about the input string makes it unparsable
    to the parser. This a somewhat expected exception in that it is not that something
    unexpected happened. The caller will likely just need to try a different parser
    for that string."""
    pass


class UnparsableTypeError(StringProblemParseError):
    pass


class UnparseableObjectError(StringProblemParseError):
    pass


def SingleTypeImplParserFunc(
    run: TypeParserRun,
    string: str,
    result: TypeParserResult
) -> None:
    """The parser function for a special builtin TypeParser which works when
    there is only one implementation of the specified type"""
    num_impls = len(run.all_type_implementations)
    if num_impls != 1:
        raise AInixParseError(f"{run.parser_name} expects exactly one "
                              f"implementation. Found {num_impls} implementations.")
    result.set_valid_implementation(run.all_type_implementations[0])
    result.set_next_slice(0, len(string))
    

def NoArgsObjectParseFunc(
    run: ObjectParserRun,
    string: str,
    result: ObjectParserResult
) -> None:
    """A special builtin parser for objects with no args. Doesn't need to
    really do anything..."""
    pass


def NoArgsObjectToStringFunc(
    args: ObjectNodeArgMap,
    result: ObjectToStringResult
) -> None:
    """A special builtin unparser for objects with no args. Doesn't need to
    really do anything..."""
    return
