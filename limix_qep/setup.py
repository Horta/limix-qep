def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('limix_qep', parent_package, top_path)
    config.add_subpackage('lik')
    config.add_subpackage('test')
    config.add_subpackage('tool')
    config.add_subpackage('ep')
    config.add_subpackage('special')
    config.make_config_py() # installs __config__.py
    return config
