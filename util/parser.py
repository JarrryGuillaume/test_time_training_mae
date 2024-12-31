import types

def load_args_from_file(filepath):
    """
    Loads variables from a Python file and returns an argparse-like Namespace object.

    Args:
        filepath (str): The path to the Python file containing variables.

    Returns:
        types.SimpleNamespace: An object with the variables as attributes.
    """
    namespace = types.SimpleNamespace()
    try:
        with open(filepath, "r") as file:
            code = compile(file.read(), filepath, 'exec')
            exec(code, {}, namespace.__dict__)
    except Exception as e:
        print(f"Error loading file {filepath}: {e}")
    return namespace