"""
Returns a List of floats representing measured values of custom features.
This List eventually gets appended to OpenAI's embedding of the poem.

"""

from typing import List

def get_custom_features(self, input: str) -> List[float]:
    # call all the functions in order
    features = List[float]
