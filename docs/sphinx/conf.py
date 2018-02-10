import subprocess

subprocess.call('cd .. ; doxygen', shell=True)
html_extra_path = ['../html/html']
master_doc = 'index'
source_suffix = '.rst'
