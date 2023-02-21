import os


def get_files(parent_dir,wanted_file=None,pattern=None,not_field=None):
    all_files = []
    all_dirs = os.listdir(parent_dir)
    dirs = []
    #
    for d in all_dirs:
        if os.path.isdir(os.path.join(parent_dir,d)):
            if pattern is None:
                dirs.append(os.path.join(parent_dir,d))
            elif isinstance(pattern,list):
                if not_field is not None:
                    if all(p in d for p in pattern) and not_field not in d:
                        dirs.append(os.path.join(parent_dir,d))
                else:
                    if all(p in d for p in pattern):
                        dirs.append(os.path.join(parent_dir,d))
            else:
                if not_field is not None:
                    if pattern in d and not_field not in d:
                        dirs.append(os.path.join(parent_dir,d))
                else:
                    if pattern in d:
                        dirs.append(os.path.join(parent_dir,d))
    #get files
    for d in dirs:
        files = os.listdir(d)
        all_files.extend(files)
    result_files = []

    if wanted_file is not None:
        for individual_file in all_files:
            if wanted_file in individual_file:
                result_files.append(individual_file)
    else:
        result_files=all_files

    paths = [f"{p}/{f}" for f,p in zip(result_files,dirs)]
    return paths