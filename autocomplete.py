# Write function to autocomplete a sentence given prefix
# Idea:
# - tree search if prefix is in corpus,
# - RNN model if we have not seen exact prefix before

def autocomplete(prefix):
    """ TODO: add helpful comments """

    if isinstance(prefix, list):
        prefix.append("as modified in autocomplete")
    return prefix
