# @Author: Tang Yubin <tangyubin>
# @Date:   2019-05-26T12:29:46+08:00
# @Email:  tang-yu-bin@qq.com
# @Last modified by:   tangyubin
# @Last modified time: 2019-05-26T16:28:11+08:00
rm -rf build
rm -rf dist
rm -rf tidyplot.egg-info

python3 -m pip install --user --upgrade setuptools wheel
python3 setup.py sdist bdist_wheel

python3 -m pip install --user --upgrade twine
python3 -m twine upload dist/*
