# import inspect
# import json
# import random
# import string

# import requests


# class FuncInspector:
#   def __init__(self):
#     self.calls = []


#   def inspect_inner(self, func, function_calls):
#     def wrapper(*args, **kwargs):
#       result = func(*args, **kwargs)
#       self.merge_result(result, function_calls)
#       return result
#     return wrapper


#   def inspect_func(self, func):
#     def wrapper(*args, **kwargs):
#       result = func(*args, **kwargs)
#       stack = inspect.stack()
#       function_calls = []
#       for i in range(len(stack)):
#         if stack[i][3] == "<module>":
#           break
#         if stack[i][3] != "wrapper":
#           function_calls.append(stack[i][3])
#       function_calls.reverse()
#       result = self.inspect_inner(result, function_calls)
#       return result
#     return wrapper
  
  
#   def parse(self, obj, delete_empty=False):
#     if isinstance(obj, list):
#       for elem in obj:
#         self.parse(elem, delete_empty)
#     if isinstance(obj, dict):
#       to_delete = []
#       for key in obj:
#         if delete_empty and not obj[key] or key == "completions":
#           to_delete.append(key)
#         else:
#           self.parse(obj[key], delete_empty)
#       for key in to_delete:
#         obj.pop(key)


#   def merge_result(self, result, function_calls):
#     prev_list = self.calls
#     prev_call = {} if not prev_list else prev_list[-1]
#     for call in function_calls[:-1]:
#       if call not in prev_call:
#         prev_call = {call: []}
#         prev_list.append(prev_call)
#       prev_list = prev_call[call]
#       prev_call = {} if not prev_list else prev_list[-1]

#     example_obj = result[0]
#     self.parse(example_obj)
#     prev_list.append({ function_calls[-1]: example_obj })


#   def view_data(self):
#     chars = string.digits + string.ascii_lowercase
#     id = ''.join(random.choices(chars, k=8))

#     post_url = 'http://127.0.0.1:5000/log-item'
#     parsed_calls = self.calls.copy()
#     self.parse(parsed_calls, delete_empty=True)
#     data = {'id': id, 'content': parsed_calls}
#     response = requests.post(post_url, json=data)
    
#     if response.status_code == 201:
#       print('Data created successfully')
#     else:
#       print(f'Error sending data to server: {response.status_code}')
#       return

#     frontend_url = f"http://localhost:3000?id={id}"
#     print(f"View the data here, {frontend_url}")


#   def output_json(self, out_path):
#     f = open(out_path, "w")
#     json_object = json.dumps(self.calls, indent=2)
#     f.write(json_object)
