# https://www.w3resource.com/python-exercises/decorator/python-decorator-exercise-7.php

import time
def rate_limits(max_calls, period):
    def decorator(func):
        calls = 0
        last_reset = time.time()

        def wrapper(*args, **kwargs):
            nonlocal calls, last_reset

            # Calculate time elapsed since last reset
            elapsed = time.time() - last_reset

            # If elapsed time is greater than the period, reset the call count
            if elapsed > period:
                calls = 0
                last_reset = time.time()

            # Check if the call count has reached the maximum limit
            if calls >= max_calls:
                raise Exception("Rate limit exceeded. Please try again later.")

            # Increment the call count
            calls += 1

            # Call the original function
            return func(*args, **kwargs)

        return wrapper
    return decorator