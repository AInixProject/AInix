import string
import random


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    # ref https://stackoverflow.com/questions/2257441/
    return ''.join(random.choice(chars) for _ in range(size))
