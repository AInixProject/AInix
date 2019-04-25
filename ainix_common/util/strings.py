import string
import random


def id_generator(size=6, chars=string.ascii_uppercase + string.digits, seed=None):
    # ref https://stackoverflow.com/questions/2257441/
    if seed:
        rng = random.Random(seed)
        return ''.join(rng.choice(chars) for _ in range(size))
    else:
        return ''.join(random.choice(chars) for _ in range(size))
