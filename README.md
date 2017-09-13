# Final Project Repository Template

This is the final project repository template for
[Machine Learning with Probabilistic Programming](http://www.proditus.com/syllabus.html).

## Duplicating your own copy of this repository

Please follow these
[instructions](https://help.github.com/articles/duplicating-a-repository/)
to make a copy of this repository and push it to your own GitHub account.

Make sure to create a **new repository** on your own GitHub account before
starting this process.

## Final Project Notebook
We have included a example of a Jupyter notebook under
`/notebook-example/example.ipynb`. This shows how to use markdown along with
LaTeX to create section headings and typeset math.

Your final project notebook should go under
`/final-project/final-notebook.ipynb`. This notebook will be your final report.
We must be able to run it in a reasonable amount of time. (If your project
involves a massive dataset, please contact me.)

Your final report should be 8 pages long. Since it is hard to translate between
a Jupyter notebook and page numbers, we've come up with the following metric:
> the Markdown export of your notebook should be approximately 1500 words.

To compute this, save your Jupyter notebook as a Markdown file by going to
```
File > Download as > Markdown (.md)
```
and then counting the words
```
wc -w final-notebook.md
```

Since this includes your code as well, we encourage you to develop separate
python scripts and include these in your final notebook. My recommendation is
that you only do basic data loading, manipulation, and plotting within Jupyter;
do all of the heavy lifting in separate Python files. (Note our strict
guidelines on coding style below.)

### Structure
Your notebook should follow the basic structure described in the project
proposal template. Make sure to clearly indicate section headings and to
present a clear narrative structure. Every subsection of your report should
correspond to a particular step of Box's loop. Feel free to include images; you
can embed them in markdown cells.

## Development
Use Python 3.6+. (I use Python 3.6.1).

Configure a virtual environment.
Follow the documentation
[here](https://docs.python.org/3.6/tutorial/venv.html).
(I like to use [virtualenvwrapper](http://virtualenvwrapper.readthedocs.io/).)

Once you activate the virtual environment, use `pip` to install a variety of
packages.
```{bash}
(venv)$ pip install -r requirements.txt
```

This should install Edward, along with Jupyter and other useful libraries.
You should see a message at the end that resembles something like
```
Successfully installed appnope-0.1.0 bleach-1.5.0 cycler-0.10.0 decorator-4.1.2 edward-1.3.3 entrypoints-0.2.3 flake8-3.4.1 html5lib-0.9999999 ipykernel-4.6.1 ipython-6.1.0 ipython-genutils-0.2.0 ipywidgets-7.0.0 jedi-0.10.2 jinja2-2.9.6 jsonschema-2.6.0 jupyter-1.0.0 jupyter-client-5.1.0 jupyter-console-5.2.0 jupyter-core-4.3.0 markdown-2.6.9 markupsafe-1.0 matplotlib-2.0.2 mccabe-0.6.1 mistune-0.7.4 nbconvert-5.3.1 nbformat-4.4.0 notebook-5.0.0 numpy-1.13.1 olefile-0.44 pandas-0.20.3 pandocfilters-1.4.2 pexpect-4.2.1 pickleshare-0.7.4 pillow-4.2.1 prompt-toolkit-1.0.15 protobuf-3.4.0 ptyprocess-0.5.2 py-1.4.34 pycodestyle-2.3.1 pyflakes-1.5.0 pygments-2.2.0 pyparsing-2.2.0 pytest-3.2.2 pytest-flake8-0.8.1 pytest-runner-2.12.1 python-dateutil-2.6.1 pytz-2017.2 pyzmq-16.0.2 qtconsole-4.3.1 scipy-0.19.1 seaborn-0.8.1 simplegeneric-0.8.1 six-1.10.0 tensorflow-1.3.0 tensorflow-tensorboard-0.1.6 terminado-0.6 testpath-0.3.1 tornado-4.5.2 traitlets-4.3.2 wcwidth-0.1.7 werkzeug-0.12.2 widgetsnbextension-3.0.2
```

### Additional dependencies
If you introduce any new dependencies to your final project, you **MUST**
update `requirements.txt` with pinned versioning.

If you plan to introduce any non-pip-installable (e.g. Stan) dependencies to
your final project, you **MUST** provide a `Dockerfile`. (Please contact me before you do so.)

### Git stuff
There is a comprehensive `.gitignore` file in this repository. This should prevent you from committing any unnecessary files. Please edit it as needed and do not commit any large files to the repository. (Especially huge datasets.)

### Code styling
Any additional code you write must pass `flake8` linting. See this
[blog post](https://medium.com/python-pandemonium/what-is-flake8-and-why-we-should-use-it-b89bd78073f2) for details.

The first thing we will do after cloning your repository is:
```{bash}
(venv)$ pytest --flake8
```

If your repository fails any checks, we will **deduct 20%** from your final project grade.
