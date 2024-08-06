from typing import Dict
from inspect import signature
import os
import shutil


def validate_params(obj: str) -> Dict[str, str]:
    """
    Return a validated dictionary of a object's parameters.

    Arguments:
        obj (object): The object to represent.
        
    Returns:
        dict: Validated dictionary the object's parameters.
        
    Raises:
        NotImplementedError: If the class does not implement `get_parameters` method.
        TypeError: If `get_parameters` does not return a dictionary with string keys.
    """
    
    # Check that the Class has a `get_parameters` method.
    if not hasattr(obj, 'get_parameters') or not callable(getattr(obj, 'get_parameters')):
        raise NotImplementedError(f"[ERROR] {obj.__class__.__name__}: Must implement a 'get_parameters' method.")
    
    # Retrieve object parameters
    params = obj.get_parameters()
    
    # Check that parameters are stored in a dictionary
    if not isinstance(params, dict):
        raise TypeError(f"[ERROR] {obj.__class__.__name__}: The 'get_parameters' method must return a dictionary. Got {type(params).__name__}.")
    
    # Check and validate parameter values so they are all strings prepared for string representation of an object
    param_str_dict = {}
    for key, value in params.items():
        if not isinstance(key, str):
            raise TypeError(f"[ERROR] {obj.__class__.__name__}: All keys in the 'get_parameters' dictionary must be strings. Got {type(key).__name__} for key: {key}.")
        
        # If a value is not a string or is not coercible to a string, then set it as "Unreadable"
        try:
            value_str = str(value)
        except Exception:
            value_str = "Unreadable"
            
        param_str_dict[key] = value_str
    
    return param_str_dict


def base_repr(obj: object) -> str:
    """
    Return a string representation of a Class object for debugging.

    Arguments:
        obj (object): The object to represent.
        
    Returns:
        str: String representation of the object.
    """
    
    params = validate_params(obj=obj)
    
    param_str_list = []
    for key, value in params.items():
        key_value = f"{key}={value}"
        param_str_list.append(key_value)

    # Combine parameter values into a string representation for the object
    param_str = ', '.join(param_str_list)
    result = f"{obj.__class__.__name__}({param_str})"
    
    return result


def base_str(obj: object) -> str:
    """
    Return a string representation of a Class object for the end user.

    Arguments:
        obj (object): The object to represent.
        
    Returns:
        str: User-friendly string representation of the object.
    """
    
    params = validate_params(obj=obj)
    
    # Combine parameter values into a string representation for the object
    param_str_list = []
    for key, value in params.items():
        key_value = f"{key.replace('_', ' ').capitalize()}: {value}"
        param_str_list.append(key_value)
        
    result = '\n'.join(param_str_list)
    
    return result


def gateway(name: str, args: dict, functions: dict, mode: str = "run"):
    """
    A gateway function for running a function from a homogenous set of functions.

    Args:
        name (str): The name of the function to run.
        args (dict): The input arguments for the function.
        functions (dict): The dictionary of available functions
        mode (str, optional): The gateway mode to check/run. Defaults to "run".

    Raises:
        ValueError: If name is not available, required arg is missing, arg is invalid, or mode is not check or run.
        TypeError: If name, mode, argument names, or function names are not string.

    Returns:
        The selected function's output.
    """
    
    if not isinstance(name, str):
        raise TypeError(f"[ERROR] Gateway: Name must be a string. Got {type(name).__name__}.")
    if not isinstance(mode, str):
        raise TypeError(f"[ERROR] Gateway: Mode must be a string. Got {type(mode).__name__}.")
    
    name = name.lower()
    mode = mode.lower()
    
    if not all([isinstance(i, str) for i in args.keys()]):
        temp = ', '.join(['(' + str(i) + ', ' + type(i).__name__ + ')' for i in args.keys() if not isinstance(i, str)])
        raise TypeError(f"[ERROR] Gateway: All argument names must be a string. Got {temp}.")
    if not all([isinstance(i, str) for i in functions.keys()]):
        temp = ', '.join(['(' + str(i) + ', ' + type(i).__name__ + ')' for i in functions.keys() if not isinstance(i, str)])
        raise TypeError(f"[ERROR] Gateway: All function names must be a string. Got {temp}.")
    
    if mode not in ["check", "run"]:
        raise ValueError(f"[ERROR] Gateway: Mode must be one of [check, run]. Got {mode}.")
    
    # Check if function name is available
    if name not in functions.keys():
        if mode == "run":
            raise ValueError(f"[ERROR] Gateway: Name was not found in list of available functions. Got {name}. Available functions: {', '.join(functions.keys())}.")
        elif mode == "check":
            return False
        
    # Function is available
    if mode == "check":
        return True
    
    # Get function from list of available functions
    fn = functions[name]
    
    # Get function arguments from its signature
    fn_signature = signature(fn)
    fn_params = fn_signature.parameters
    
    # Get list of required arguments
    required_args = [param for param, details in fn_params.items() if details.default == details.empty]
    
    # Check if all required arguments are provided
    missing_args = []
    for req_arg in required_args:
        if req_arg not in args.keys():
            missing_args.append(req_arg)
            
    if len(missing_args) > 1:
        raise ValueError(f"[ERROR] Gateway: Missing required arguments: {', '.join(missing_args)}.")
    
    # Check if all provided argument names are valid
    invalid_args = []
    for arg in args.keys():
        if arg not in fn_params:
            invalid_args.append(arg)
            
    if len(invalid_args) > 1:
        raise ValueError(f"[ERROR] Gateway: Invalid argument names: {', '.join(invalid_args)}.")
    
    # Run function with argument inputs
    output = fn(**args)
    
    return output


def manage_directory(directory_path: str, delete_if_exists: bool = False) -> None:
    """
    Checks if a directory exists. If it does, deletes and recreates directory or provides written 
    warning of potential collision. If it doesn't, then creates directory at path.

    Args:
        directory_path (str): Filepath to new directory.
        delete_if_exists (bool): Option to be conservative with deletion capability.

    Raises:
        ValueError: _description_
    """
    
    # Check if directory exists
    if os.path.exists(directory_path):
        # Check if path is a directory
        if os.path.isdir(directory_path):
            if delete_if_exists:
                # Delete the old directory and all of its contents
                shutil.rmtree(directory_path)
                print(f"[UPDATE] Manage Directory: Deleted existing directory: {directory_path}")
            else:
                # Warn user of collision in conservative mode
                print(f"[WARNING] Manage Directory: The directory {directory_path} already exists and will not be deleted.")
                return
        else:
            raise ValueError(f"[ERROR] Manage Directory: The path {directory_path} exists but is not a directory.")
        
    # Create the directory
    os.makedirs(directory_path)
    print(f"[UPDATE] Manage Directory: Created directory: {directory_path}")