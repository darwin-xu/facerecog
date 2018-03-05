#!/usr/bin/env python
"""Send classify request to server."""
import requests


def classify():
    """classify file."""
    base_uri = 'http://127.0.0.1:5000'
    # Sent classify request
    url_classify = base_uri + "/classifyFace"

    content = {}

    response = requests.post(url_classify, data=content)
    if response.ok:
        print("Start classifying...")
    else:
        print(response)


classify()
