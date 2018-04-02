#!/usr/bin/env python
"""Send classify request to server."""
import requests
from param import base_uri


def classify():
    """classify file."""
    # Sent classify request
    url_classify = base_uri + "/classifyFace"

    content = {}

    response = requests.post(url_classify, data=content)
    if response.ok:
        print("Start classifying...")
    else:
        print(response)


classify()
