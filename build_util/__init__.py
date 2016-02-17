import os
import sys
import subprocess

# Return the git revision as a string
def _git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        cwd = os.path.abspath(os.path.dirname(__file__))
        cwd = os.path.join(cwd, '../')
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                               env=env, cwd=cwd).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION

def get_version_info(version, isreleased, filename):
    # Adding the git rev number needs to be done inside write_version_py(),
    # otherwise the import of numpy.version messes up the build under Python 3.
    FULLVERSION = version
    cwd = os.path.abspath(os.path.dirname(__file__))
    if os.path.exists(os.path.join(cwd, '../', '.git')):
        GIT_REVISION = _git_version()
    elif os.path.exists(filename):
        # must be a source distribution, use existing version file
        try:
            from limix_qep.version import git_revision as GIT_REVISION
        except ImportError:
            msg = "Unable to import git_revision. Try removing "
            msg += "%s and the build directory " % filename
            msg += "before building."
            print('Working folder %s' % os.getcwd())
            raise ImportError(msg)
    else:
        GIT_REVISION = "Unknown"

    if not isreleased:
        FULLVERSION += '.dev0+' + GIT_REVISION[:7]

    return FULLVERSION, GIT_REVISION

def write_version_py(version, isreleased, filename):
    cnt = """
# THIS FILE IS GENERATED FROM LIMIX-QEP SETUP.PY
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
release = %(isrelease)s
if not release:
    version = full_version
"""
    FULLVERSION, GIT_REVISION = get_version_info(version, isreleased, filename)

    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': version,
                       'full_version' : FULLVERSION,
                       'git_revision' : GIT_REVISION,
                       'isrelease': str(isreleased)})
    finally:
        a.close()

def generate_cython():
    cwd = os.path.abspath(os.path.dirname(__file__))
    print("Cythonizing sources")
    p = subprocess.call([sys.executable,
                          os.path.join(cwd, 'cythonize.py'),
                          'limix_qep/special'])
    if p != 0:
        raise RuntimeError("Running cythonize failed!")
