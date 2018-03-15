class ls(AIProgram):
    def get_definition():
        my_definition = AIProgramDefinition()
        my_definition.add_argument(StoreTrueArgument(name = "-a"))
        my_definition.add_argument(StoreTrueArgument(name = "-l"))
        
        my_definition.add_argument(PositionalArgument(List(Path()))
        return my_definition
    pass
    
if __name__ == "__main__":
    ainix.autoinvoke(ls)
