class AInixType:
    """Used to contruct AInix types.

    Types represent a collection of related objects.

    Args:
        type_name : An upper CamelCase string to identify this type
    """
    def __init__(self, name : str, default_value_parser = None):
        self.name = name
        self.default_parser = default_value_parser

class AInixObject:
    def __init__(self, 
        name : str, 
        type : AInixType, 
        children : list,
        direct_sibling
    ):
        self.name = name

class ValueParser():
    def __init__(self, type : AInixType):
        self.type = type

class StructuralParser():
    pass

class AInixArgument:
    def __init__(self, 
        type : AInixType, 
        value_parser : ValueParser = None,
        required : bool = False,
        parse_data : dict = {}
    ):
        self.type = type
        self.required = required
        if ValueParser is not None:
            self.value_parser = value_parser
        else:
            if type.default_parser is None:
                raise ValueError("No value_parser provided for an AInixArgument. \
                    However, type %s does not provide a default value_parser" \
                    % (self.type.name,))
            self.value_parser = type.value_parser
        self.parse_data = parse_data
