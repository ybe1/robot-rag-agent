import os

def get_project_root()->str:
    current_file=os.path.abspath(__file__)

    current_dir=os.path.dirname(current_file)

    project_dir=os.path.dirname(current_dir)

    return project_dir

def get_abs_path(relative_path:str)->str:
    project_root=get_project_root()
    return os.path.join(project_root,relative_path)

if __name__ == '__main__':
    root=get_project_root()
    print(root)

    abs_path=get_abs_path('utils/path_tool.py')
    print(abs_path)