"""Simple util funcs for testing"""


def send_result(gen, send_value):
    """Get the result from a generator when expected to return next."""
    try:
        gen.send(send_value)
    except StopIteration as stp:
        return stp.value
    raise ValueError("Expected it to stop")