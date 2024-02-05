
import inspect
import textwrap
import ast 


def inspect_request_keys(func):
    code = inspect.getsource(func)  # Get the source code of the class
    code = textwrap.dedent(code)  # remove any indentation

    variable_name = "data"
    
    # Parse the code snippet into an Abstract Syntax Tree (AST)
    tree = ast.parse(code)

    # Create a list to store the extracted attributes
    request_keys = []

    # Define a function to recursively traverse the AST and extract attributes
    def extract_attributes_recursive(node, current_attribute):
        if isinstance(node, ast.Attribute):
            # If the node is an attribute, update current_attribute and recursively traverse
            current_attribute = node.attr + "." + current_attribute
            extract_attributes_recursive(node.value, current_attribute)
        elif isinstance(node, ast.Name) and node.id == variable_name:
            # If the node is a Name node with the variable name, add it to the attributes list
            request_keys.append(current_attribute[:-1])  # Remove the trailing dot
        else:
            # Recursively traverse child nodes
            for child_node in ast.iter_child_nodes(node):
                extract_attributes_recursive(child_node, current_attribute)

    # Start the extraction process
    extract_attributes_recursive(tree, "")

    # Filter out attributes that are actually method calls
    request_keys = [attr for attr in request_keys if not code.count(attr + "(")]

    return request_keys