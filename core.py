def __init__(extractor, classifier, aggregator, threshold):

def validate(response, context, prompt):
    if not response:
        raise ValueError("Empty response is not usable to infer")
    
    if not context:
        raise ValueError("Empty context is not usable to infer")
    
    