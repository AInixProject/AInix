from argparse import ArgumentParser
from collections import defaultdict
from typing import Tuple, List, Dict
import attr

from ainix_common.parsing.loader import TypeContextDataLoader
from ainix_common.parsing.parse_primitives import AInixParseError
from ainix_common.parsing.stringparser import StringParser
from ainix_common.parsing.typecontext import TypeContext
from ainix_kernel.training.train_contexts import ALL_EXAMPLE_NAMES, load_all_examples
from ainix_kernel.specialtypes import allspecials
import yaml
try:
    from yaml import CLoader as _YamlLoader, CDumper as Dumper
except ImportError:
    from yaml import Loader as _YamlLoader, Dumper


def get_all_data(nl_path: str, cm_path: str) -> Tuple[str, str]:
    data = []
    with open(nl_path) as nlf:
        with open(cm_path) as cmf:
            for nl, cm in zip(nlf, cmf):
                data.append((nl.strip(), cm.strip()))
    return data


def get_a_tc() -> TypeContext:
    type_context = TypeContext()
    loader = TypeContextDataLoader(type_context, up_search_limit=4)
    loader.load_path("builtin_types/generic_parsers.ainix.yaml")
    loader.load_path("builtin_types/command.ainix.yaml")
    loader.load_path("builtin_types/paths.ainix.yaml")
    allspecials.load_all_special_types(type_context)
    for f in ALL_EXAMPLE_NAMES:
        loader.load_path(f"builtin_types/{f}.ainix.yaml")
    type_context.finalize_data()
    return type_context


def get_parsable_commands(data: Tuple[str, str]) -> List[Tuple[str, str]]:
    tc = get_a_tc()
    parser = StringParser(tc)
    parsable_data = []
    for nl, cm in data:
        try:
            ast = parser.create_parse_tree(cm, "CommandSequence")
            parsable_data.append((nl, cm))
            print(f"PASS {cm}")
        except AInixParseError as e:
            pass
    return parsable_data


def merge_command_sets(examples: List[Tuple[str, str]]):
    # Make a graph of all the examples.
    # x_nodes maps from an x string to all y strings that are valid
    # y_nodes maps from a y string to all x strings that are valid
    x_nodes: Dict[str, List[str]] = defaultdict(list)
    y_nodes: Dict[str, List[str]] = defaultdict(list)
    for nl, cm in examples:
        x_nodes[nl].append(cm)
        y_nodes[cm].append(nl)
    # Go through all the x nodes pulling everything it is connected to into
    # an example pair. Remove all those from the remain graph, and continue
    # until out of nodes.
    valid_maps: List[Tuple[List[str], List[str]]] = []
    while x_nodes:
        # Setup for exploring everything connected to a node
        xs_need_to_visit: List[str] = [next(iter(x_nodes))]
        ys_need_to_visit: List[str] = []
        so_far_valid_xs, so_far_valid_ys = set(), set()
        # Loop through adding everything connected to this first x
        while xs_need_to_visit or ys_need_to_visit:
            if xs_need_to_visit:
                new_x = xs_need_to_visit.pop()
                so_far_valid_xs.add(new_x)
                ys_need_to_visit.extend(x_nodes.pop(new_x, []))
            if ys_need_to_visit:
                new_y = ys_need_to_visit.pop()
                so_far_valid_ys.add(new_y)
                xs_need_to_visit.extend(y_nodes.pop(new_y, []))
        valid_maps.append((sorted(list(so_far_valid_xs)), sorted(list(so_far_valid_ys))))
    return valid_maps


def convert_to_exampleset_dict(valid_maps, y_type: str = "CommandSequence") -> dict:
    return {
       "defines": [{
           "define_new": "example_set",
           "y_type": y_type,
           "examples": [
               {"x": xs, "y": ys}
               for xs, ys in valid_maps
           ]
       }]
    }


def get_yaml_str(example_dict: dict):
    return yaml.dump(example_dict, default_flow_style=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--nl_source", type=str, default="./data/all.nl.filtered")
    parser.add_argument("--cm_source", type=str, default="./data/all.cm.filtered")
    parser.add_argument("--out_file", type=str, default="./tellina.ainix.yaml")
    args = parser.parse_args()

    data = get_all_data(args.nl_source, args.cm_source)
    parsable_examples = get_parsable_commands(data)
    print(f"Found {len(parsable_examples)} examples")
    valid_maps = merge_command_sets(parsable_examples)
    example_dict = convert_to_exampleset_dict(valid_maps)
    yampl_str = get_yaml_str(example_dict)
    if args.out_file:
        print(f"Writing to {args.out_file}")
        with open(args.out_file, "w") as f:
            f.write(yampl_str)




