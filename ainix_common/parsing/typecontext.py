from collections import defaultdict
from typing import List, Optional, Dict
import ainix_common.parsing.parse_primitives
from ainix_common.util.strings import id_generator
SINGLE_TYPE_IMPL_BUILTIN = "SingleTypeImplParser"


class AInixType:
    """Used to construct AInix types.

    Types represent a collection of related objects. From a modeling perspective,
    this defines a classification problem between all implementations (objects)
    that you later register.

    Args:
        type_context : the type context to register this type in. Registration
            is done automatically for you on construction.
        name : An upper CamelCase string to identify this type
        default_type_parser_name : string identifier of the parser to use for
            this type by default. An argument can request specify that different
            type parser should be used instead of the default. If "None" is
            given then arguments are required to specify a type parser.
        default_object_parser_name : String identifier of the ObjectParser
            to use when parsing implementations of this type. This can
            be changed by the implementation itself or potentially by the
            TypeParser that parsed this type. Specifying "None" means that
            implementations are required to give a parser to use.
        allowed_attributes : TODO (DNGros)
    """
    def __init__(
        self,
        type_context: 'TypeContext',
        name: str,
        default_type_parser_name: Optional[str] = None,
        default_object_parser_name: Optional[str] = None,
        allowed_attributes: Optional[List[str]] = None
    ):
        self._type_context = type_context
        self.name = name
        self.default_type_parser_name = default_type_parser_name
        self.default_object_parser_name = default_object_parser_name
        self.allowed_attributes = allowed_attributes if allowed_attributes else []
        type_context.register_type(self)

    @property
    def default_type_parser(self) -> Optional['ainix_common.parsing.parse_primitives.TypeParser']:
        if self.default_type_parser_name is None:
            return None
        retrieved_parser = self._type_context.get_type_parser_by_name(
            self.default_type_parser_name)
        if retrieved_parser is None:
            raise RuntimeError(f"{self} unable to retrieve default_type_parser "
                               f"{self.default_type_parser_name} from current "
                               f"run context")
        return retrieved_parser

    @property
    def default_object_parser(self) \
            -> Optional['ainix_common.parsing.parse_primitives.ObjectParser']:
        if self.default_object_parser_name is None:
            return None
        retrieved_parser = self._type_context.get_object_parser_by_name(
            self.default_object_parser_name)
        if retrieved_parser is None:
            raise RuntimeError(f"{self} unable to retrieve default_object_parser "
                               f"{self.default_object_parser_name} from current "
                               f"run context")
        return retrieved_parser

    @property
    def type_context(self):
        return self._type_context

    def __str__(self):
        return f"<AInixType: {self.name}>"

    def __lt__(self, other):
        return self.name < other.name

    def __gt__(self, other):
        return self.name > other.name


class AInixObject:
    def __init__(
        self,
        type_context: 'TypeContext',
        name: str,
        type_name: str,
        children: List['AInixArgument'] = None,
        preferred_object_parser_name: Optional[str] = None,
        type_data: Optional[dict] = None
    ):
        self._type_context = type_context
        self.name = name
        if type_name is None:
            raise ValueError(f"AInixObject {name} must have a non-None type_name")
        self.type_name = type_name
        self.children: List['AInixArgument'] = children if children else []
        self.arg_name_to_index = {arg.name: i for i, arg in enumerate(self.children)}
        self.type_data = type_data
        self.preferred_object_parser_name = preferred_object_parser_name
        self._type_context.register_object(self)

    @property
    def type(self) -> AInixType:
        return self._type_context.get_type_by_name(self.type_name)

    @property
    def type_context(self) -> 'TypeContext':
        return self._type_context

    def get_arg_by_name(self, arg_name: str):
        if arg_name not in self.arg_name_to_index:
            return None
        return self.children[self.arg_name_to_index[arg_name]]

    @property
    def preferred_object_parser(self) \
            -> Optional['ainix_common.parsing.parse_primitives.ObjectParser']:
        """Returns the instance associated with the preferred_object_parser_name"""
        if self.preferred_object_parser_name is None:
            return None
        lookup_parser = self._type_context.get_object_parser_by_name(
            self.preferred_object_parser_name)
        if lookup_parser is None:
            raise RuntimeError(f'{self} unable to get preferred_object_parser of name '
                               f'"{self.preferred_object_parser_name}" in '
                               f'current context.')
        return lookup_parser

    def __repr__(self):
        return f"<AInixObject: {self.name}>"

    def __lt__(self, other):
        return self.name < other.name

    def __gt__(self, other):
        return self.name > other.name


# This is the arg of the dummy object for an arg is created
OPTIONAL_ARGUMENT_NEXT_ARG_NAME = "__ARG_PRESENT_NEXT"


class AInixArgument:
    def __init__(
        self,
        type_context: 'TypeContext',
        name: str,
        type_name: Optional[str],
        type_parser_name: Optional[str] = None,
        required: bool = False,
        arg_data: dict = None
    ):
        self._type_context = type_context
        self.name = name
        self.type_name = type_name
        self.required = required
        self.type_parser_name = type_parser_name
        self.arg_data = arg_data if arg_data else {}
        if required and type_name is None:
            raise ValueError(f"Arg {name} is required but None type.")
        self._create_optional_args_types()

    def _create_optional_args_types(self):
        if not self.required:
            self.present_choice_type = AInixType(
                self.type_context, self._make_optional_arg_type_name(), None, None, None
            )
            is_present_name = f"{self.present_choice_type.name}.PRESENT"
            if self.type_name is not None:
                present_args = [AInixArgument(
                    self.type_context, OPTIONAL_ARGUMENT_NEXT_ARG_NAME,
                    self.type_name, None, True
                )]
            else:
                present_args = []
            self.is_present_object = AInixObject(
                self.type_context, is_present_name,
                self.present_choice_type.name, present_args
            )
            not_present_name = f"{self.present_choice_type.name}.NOTPRESENT"
            self.not_present_object = AInixObject(
                self.type_context, not_present_name, self.present_choice_type.name
            )
        else:
            self.present_choice_type = None
            self.is_present_object = None
            self.not_present_object = None

    def _make_optional_arg_type_name(self) -> str:
        return f"__arg_present_choice_type.{id_generator(5)}.{self.name}"

    @property
    def type(self) -> Optional[AInixType]:
        if self.type_name is None:
            return None
        return self._type_context.get_type_by_name(self.type_name)

    @property
    def next_choice_type(self) -> Optional[AInixType]:
        """Gets the type of the argument, taking into acount if the arguement
        is optional or not."""
        if self.required:
            return self.type
        else:
            return self.present_choice_type

    @property
    def type_context(self) -> 'TypeContext':
        return self._type_context

    @property
    def type_parser(self) -> Optional['ainix_common.parsing.parse_primitives.TypeParser']:
        if self.type_parser_name is None:
            type_default_parser = self.type.default_type_parser
            if type_default_parser is None:
                raise ValueError(f"Argument {self.name} of type {type} does"
                                 f"not have prefered parser name or default"
                                 f"type parser")
            return type_default_parser
        return self._type_context.get_type_parser_by_name(self.type_name)

    def __repr__(self):
        return f"<AInixArgument {self.name} type {self.type.name}>"


class TypeContext:
    """Used to track a set types, objects, and parsers.
    Construction of types of objects must be given a context
    in order identify and track their string identifiers, and to be
    able to query all implementations of a type while parsing
    or modeling.
    """
    def __init__(self):
        self._name_to_type: Dict[str, AInixType] = {}
        self._name_to_object: Dict[str, AInixObject] = {}
        self._name_to_type_parser: Dict[str, ainix_common.parsing.parse_primitives.TypeParser] = {}
        self._name_to_object_parser: \
            Dict[str, ainix_common.parsing.parse_primitives.ObjectParser] = {}
        self._type_name_to_implementations: Dict[str, List[AInixObject]] = \
            defaultdict(list)

    def _resolve_type(self, type):
        """Converts a string into a type object_name if needed"""
        if isinstance(type, str):
            got_type = self.get_type_by_name(type)
            if got_type is None:
                raise ValueError("Unable to find type", type)
            return got_type
        return type

    def get_type_by_name(self, name: str) -> AInixType:
        return self._name_to_type.get(name, None)

    def get_object_by_name(self, name: str) -> AInixObject:
        return self._name_to_object.get(name, None)

    def get_type_parser_by_name(self, name: str) -> \
            'ainix_common.parsing.parse_primitives.TypeParser':
        return self._name_to_type_parser.get(name, None)

    def get_object_parser_by_name(self, name: str) -> \
            'ainix_common.parsing.parse_primitives.ObjectParser':
        return self._name_to_object_parser.get(name, None)

    def get_implementations(self, type):
        type = self._resolve_type(type)
        return self._type_name_to_implementations[type.name]

    def register_type(self, new_type: AInixType) -> None:
        """Registers a type to be tracked. This should be called automatically when
        instantiating new types. This method may also mutate the new_type if
        it uses a builtin parser for one of its parsers"""
        if new_type.name in self._name_to_type:
            raise ValueError(f"Type {new_type.name} already exists")
        self._link_builtin_type_parser(new_type)
        # TODO (DNGros): need to link buildin object parsers
        self._name_to_type[new_type.name] = new_type

    def _link_builtin_type_parser(
        self,
        type_using_it: AInixType
    ) -> None:
        """Handles generation and linking of a special builtin type parser.

        Args:
            type_using_it: the AInixType that that wants to use the given builtin.
                This object may be mutated by changing its default_type_parse_name
                to appropriately link to the builtin. If its default_type_parse_name
                is not a builtin, then it is left unmodified
        """
        if type_using_it.default_type_parser_name == SINGLE_TYPE_IMPL_BUILTIN:
            self._link_single_type_impl_parser(type_using_it)

    def _link_single_type_impl_parser(self, type_to_change: AInixType):
        """Changes a type to use a SingleTypeImplParser

        Args:
            type_to_change: the type which we should change the default parser
                on to be a SingleTypeImplParser
        """
        link_name = f"__builtin.SingleTypeImplParser.{type_to_change.name}"
        if not self.get_type_parser_by_name(link_name):
            ainix_common.parsing.parse_primitives.TypeParser(
                self,
                link_name,
                ainix_common.parsing.parse_primitives.SingleTypeImplParserFunc,
                type_name=type_to_change.name
            )
        type_to_change.default_type_parser_name = link_name

    def _link_no_args_obj_parser(self, obj_to_change: AInixObject):
        """Changes an object to use a SingleTypeImplParser
        Args:
            obj_to_change: the object which to set its prefered object parser
                to a no args parser
        """
        link_name = f"__builtin.NoArgsObjectParser.{obj_to_change.name}"
        if not self.get_type_parser_by_name(link_name):
            ainix_common.parsing.parse_primitives.ObjectParser(
                self,
                link_name,
                ainix_common.parsing.parse_primitives.NoArgsObjectParseFunc,
                ainix_common.parsing.parse_primitives.NoArgsObjectToStringFunc,
                obj_to_change.type_name
            )
        obj_to_change.preferred_object_parser_name = link_name

    def register_type_parser(
        self,
        new_parser: 'ainix_common.parsing.parse_primitives.TypeParser'
    ) -> None:
        """Registers a parser to be tracked. This should be called automatically when
        instantiating new TypeParsers."""
        if new_parser.parser_name in self._name_to_type_parser:
            raise ValueError("Type parser", new_parser.parser_name, "already exists")
        self._name_to_type_parser[new_parser.parser_name] = new_parser

    def register_object(self, new_object: AInixObject) -> None:
        """Registers a object_name to be tracked. This should be called
        automatically when instantiating new objects."""
        name = new_object.name
        if name in self._name_to_object:
            raise ValueError("Object", name, "already exists")
        self._name_to_object[name] = new_object
        self._type_name_to_implementations[new_object.type_name].append(new_object)

    def register_object_parser(
        self,
        new_parser: 'ainix_common.parsing.parse_primitives.ObjectParser'
    ) -> None:
        """Registers a type to be tracked. This should be called automatically when
        instantiating new types."""
        if new_parser.parser_name in self._name_to_object_parser:
            raise ValueError("Object parser", new_parser.parser_name, "already exists")
        self._name_to_object_parser[new_parser.parser_name] = new_parser

    def fill_default_parsers(self):
        """Tries to fill defaults for any types that have None as a default parser
        and there is a valid default available. This includes the SingleTypeImplParser
        for types with only one implementation."""
        for type_ in self._name_to_type.values():
            if type_.default_type_parser_name is None:
                if len(self.get_implementations(type_)) == 1:
                    self._link_single_type_impl_parser(type_)
        for object_ in self._name_to_object.values():
            no_children = len(object_.children) == 0
            no_default = object_.preferred_object_parser_name is None and \
                object_.type.default_object_parser_name is None
            if no_children and no_default:
                self._link_no_args_obj_parser(object_)

    def verify(self):
        """After you have instantiated all the types and objects you need in this
        context, you can call this method to verify all referenced types are
        valid."""
        # TODO (DNGos): IMPLEMENT!
        # This method should check that all objects type's actually exist, and
        # notify the user if they have tried reference things they did not create.
        # It would also be nifty if it checked builtins to see if they were reasonable
        # (so for example, that things that use the SingleTypeImplParser actually
        #  only have one implementation)
        raise NotImplemented("Verify not implemented")
