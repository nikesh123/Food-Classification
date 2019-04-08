"""
This script runs the Foodology application using a development server.
"""

from os import environ
from foi_ai import app

if __name__ == '__main__':
    HOST = environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(environ.get('SERVER_PORT', '1111'))
    except ValueError:
        PORT = 1111
    app.run(HOST, PORT, debug=True)
