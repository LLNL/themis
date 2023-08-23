"""This module renames imports for Python 2/3 compatibility."""

import sys

# pylint: skip-file

if sys.version_info.major >= 3:
    basestring = str
    import socketserver
    import pickle
    import reprlib
    import http.server as http_server
    import http.client as http_client
    from collections import UserDict
    from urllib.parse import parse_qs, parse_qsl, urlencode
    try:
        from collections.abc import Mapping, Sequence
    except ImportError:
        from collections import Mapping, Sequence

    csvargs = {"newline": ""}  # arguments for opening files with the `csv` module

else:
    basestring = basestring
    import SocketServer as socketserver
    import cPickle as pickle
    import repr as reprlib
    import BaseHTTPServer as http_server
    import httplib as http_client
    from UserDict import UserDict
    from urlparse import parse_qs, parse_qsl
    from urllib import urlencode
    from collections import Mapping, Sequence

    csvargs = {}  # arguments for opening files with the `csv` module
