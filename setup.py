from setuptools import setup, find_packages, Extension

# setup() parameters - https://packaging.python.org/guides/distributing-packages-using-setuptools/
setup(
    name='mykmeanssp',
    version='0.1.0',
    author="Roee Negri and Gal Toren",
    author_email="roeenegri77@gmail.com",
    description="kmeanssp C-API",
    install_requires=['invoke'],
    packages=find_packages(),  # find_packages(where='.', exclude=())
    #    Return a list of all Python packages found within directory 'where'
    license='GPL-2',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
    ext_modules=[
        Extension(
            # the qualified name of the extension module to build
            'mykmeanssp',
            # the files to compile into our module relative to ``setup.py``
            sources=['spkmeans.c', 'spkmeansmodule.c']
        )
    ]
)


#include <stdio.h> 
#include <stdlib.h> /*memo alloc*/
#include <math.h> /*math ops*/
#include <string.h> /*string ops*/
#include <ctype.h>
#include "spkmeans.h"