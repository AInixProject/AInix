import math

from ainix_kernel.training.trainer import get_examples

if __name__ == "__main__":
    type_context, index, replacers, loader = get_examples()
    #\newcommand{\numUnixCommands}{FILL\_VAR}

    num_x_vals = index.get_num_x_values()
    num_y_sets = index.get_num_y_sets()
    num_y_vals = index.get_num_y_values()
    x_vals_per_y_set = round(num_x_vals / num_y_sets, 1)
    num_unix_commands = len(type_context.get_implementations("Program"))

    print(r"\newcommand{\numXValues}{" + str(num_x_vals) + "}")
    print(r"\newcommand{\numYSets}{" + str(num_y_sets) + "}")
    print(r"\newcommand{\numXValuesPerYSet}{" + str(x_vals_per_y_set) + "}")
    print(r"\newcommand{\numYValues}{" + str(num_y_vals) + "}")
    print(r"\newcommand{\numUnixCommands}{" + str(num_unix_commands) + "}")
