import os
import os.path as osp

def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    """check file is exist or not

    Args:
        filename (str): filename name to be checked.
        msg_tmpl (str):
    """
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


def mkdir_or_exist(dir_name, mode=0o777):
    """mkdir if not exist

    Args:
        dir_name (str): directory name to be made.
        mode (str):
    """
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)