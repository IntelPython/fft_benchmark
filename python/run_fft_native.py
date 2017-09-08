from __future__ import print_function
from platform import system

IS_WIN = system() == 'Windows'
IS_LIN = system() == 'Linux'
IS_MAC = system() == 'Darwin'
IS_UNIX = IS_LIN or IS_MAC

EXE_DIR = 'Scripts' if IS_WIN else 'bin'
EXE_EXTNSN = '.exe' if IS_WIN else ''

from os import getcwd, chdir, environ
from os.path import dirname, exists, join
from tempfile import TemporaryFile
from functools import partial
from sys import version_info, executable, exc_info
from subprocess import Popen
import numpy as np

PY_MAJOR_VER, PY_MINOR_VER = version_info.major, version_info.minor

def get_stderr_stdout(args, quiet=False, env_vars=None, exceptions=None):
    '''
    This will write STDOUT + STDERR (procured from subprocess) to STDOUT
    and list, and return that list. It will return a None in case the
    subprocess is exited at runtime.
    '''
    assert any([type(args) == list, type(args) == str]), \
              ("Incorrect args type, should be 'list', or 'str': "
               + str(args) + ' type: ' + str(type(args)))

    def print_text(f_out, f_err, dont_return=False):
        '''
        Print and return data written to STDOUT and STDERR (by subprocess).
        '''
        data_to_return = [] if not dont_return else None
        for key, file_desc in {'STDOUT': f_out, 'STDERR': f_err}.items():
            file_desc.seek(0, 2)
            if file_desc.tell() and not quiet:
                print('From {}: '.format(key))
            file_desc.seek(0)
            for line in file_desc:
                line = line.strip()
                if PY_MAJOR_VER == 3:
                    line = line.decode('utf-8')
                if not quiet:
                    print(line,)
                if not dont_return:
                    data_to_return.append(line)
        return data_to_return

    def extract_exception_name(name):
        '''
        Extract exception's name from tuple returned by exc_info()
        '''
        name = str(name[0]).split()[-1].strip()
        for remove in ['"', "'", '<', '>']:
            name = name.replace(remove, '')
        return name.split('.')[-1].strip()

    # Windows messages, which prompt user input after a subprocess crash, are halting
    # the test process. This will disable the window, an answer taken from stackoverflow:
    #   - http://stackoverflow.com/questions/5069224/handling-subprocess-crash-in-windows
    def disable_win_messages():
        if IS_WIN:
            import ctypes
            SEM_NOGPFAULTERRORBOX = 0x0002 # From MSDN
            ctypes.windll.kernel32.SetErrorMode(SEM_NOGPFAULTERRORBOX);
            subprocess_flags = 0x8000000 #win32con.CREATE_NO_WINDOW?
        else:
            subprocess_flags = 0
        return subprocess_flags

    execute_string = ' && '.join(args) if type(args) == list else args

    if not quiet:
        print('[$][' + getcwd() + ']> ' + execute_string)
        print('env = ' + str(env_vars))

    f_err, f_out, dont_return = None, None, False
    try:
        f_err, f_out = TemporaryFile(), TemporaryFile()
        ph = Popen(execute_string if IS_WIN else ['/bin/bash', '-c', execute_string],
              stdout=f_out, stderr=f_err, shell=(True if IS_WIN else False),
              env=env_vars, creationflags=disable_win_messages())
        ph.communicate()
    except:
        print('something went wrong')
        raise
        # Takes care of the condition where data while being written to 'data_to_return' is interrupted / test
        # framework running the suite is abruptly exited - handled by pybuilder. There's also an added provision
        # to ensure that the execution is deemed successful if the encountered exception is expected.
        except_name, dont_return = extract_exception_name(exc_info()), True
        if exceptions:
            for exc in exceptions:
                exc_name, plat = exc[0], exc[1]
                plat = any([plat == 'Win' and IS_WIN,
                            plat == 'Lin' and IS_LIN,
                            plat == 'Mac' and IS_MAC])
                if exc_name == except_name and plat:
                    dont_return = False
                    break
    finally:
        data_to_return = print_text(f_out, f_err, dont_return)
        for file_desc in [f_err, f_out]:
            if file_desc and not file_desc.closed:
                file_desc.close()
        return data_to_return


def update_envs(st_dict, update_dict):
    res = st_dict.copy()
    res.update(update_dict)
    return res

def native_perf_times(exec, pars, repetitions=6):
    global natives_dir
    header = ['=' * 10 + ' ' + exec + ' ' + '=' * 10]
    perfs = []
    exec_vars = update_envs(common_env_vars, pars)
    exec_vars = update_envs(exec_vars, {'S' : str(repetitions)})
    exec_path = natives_dir + exec
    #
    t = get_stderr_stdout(exec_path,
           quiet=True, env_vars = exec_vars)
    if not isinstance(t, list) or len(t) < 2:
        raise ValueError("Execution of %s returned an unexpected result" % exec_path)
    for i in range(repetitions):
        perfs.append(float(t[-1-i]))
    #
    header = header + t[:-repetitions]
    return '\n'.join(header), np.array(perfs)

def print_info(header, perf_times):
    print(header)
    a = np.asarray(perf_times)
    print('{min:0.3f}, {med:0.3f}, {max:0.3f}'.format(
        min=np.min(a), med=np.median(a), max=np.max(a)
    ))
    print("", flush=True)


if IS_WIN:
    natives_dir = '..\\native\\build\\'
else:
    natives_dir = '../native/build/'

if IS_WIN:
    prefix = dirname(executable)
else:
    prefix = dirname(dirname(executable))

print("$PREFIX = %s" % prefix)
if IS_WIN:
    # inherenting from the environment allows native code to see MKL
    common_env_vars = environ.copy()
    common_env_vars.update({'REPS' : '12'})
    params_1d = {'N' : '1200000'}
    params_2d = {'N1' : '1200', 'N2' : '1200'}
    params_3d = {'N1' : '113', 'N2' : '114', 'N3' : '115'}
else:
    common_env_vars = update_envs(environ, {
        'LD_LIBRARY_PATH' : prefix + '/lib' + ':' + environ.get('LD_LIBRARY_PATH', default=''),
        'REPS' : '16'})
    params_1d = {'N' : '5000000'}
    params_2d = {'N1' : '2500', 'N2' : '2500'}
    params_3d = {'N1' : '113', 'N2' : '214', 'N3' : '315'}

header, perf_times = native_perf_times('fft_cdp-out-c.exe', params_1d)
print_info(header, perf_times)

header, perf_times = native_perf_times('fft_cdp-in-c.exe', params_1d)
print_info(header, perf_times)

header, perf_times = native_perf_times('fft2_cdp-out-nc.exe', params_2d)
print_info(header, perf_times)

header, perf_times = native_perf_times('fft2_cdp-in-nc.exe', params_2d)
print_info(header, perf_times)

header, perf_times = native_perf_times('fft3_cdp-out-nc.exe', params_3d)
print_info(header, perf_times)

header, perf_times = native_perf_times('fft3_cdp-in-nc.exe', params_3d)
print_info(header, perf_times)
