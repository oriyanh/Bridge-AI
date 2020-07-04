FACES = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', ]
FACES_ALT = {'j': 'J', 'q': 'Q', 'k': 'K', 'a': 'A'}

SUITS = ['♠', '♥', '♦', '♣', ]
SUITS_ALT = {'S': '♠', 'H': '♥', 'D': '♦', 'C': '♣',
             's': '♠', 'h': '♥', 'd': '♦', 'c': '♣', }


class Card:
    """
    A playing card.
    """
    def __init__(self, face: str, suit: str):
        if face in FACES:
            self.face = face
        elif face in FACES_ALT:
            self.face = FACES_ALT[face]
        else:
            raise ValueError

        if suit in SUITS:
            self.suit = suit
        elif suit in SUITS_ALT:
            self.suit = SUITS_ALT[suit]
        else:
            raise ValueError

    def __repr__(self):
        return self.face + self.suit

    def __lt__(self, other):
        if self.face != other.face:
            return FACES.index(self.face) < FACES.index(other.face)
        else:
            return SUITS.index(self.suit) > SUITS.index(other.suit)

    def __le__(self, other):
        if self.face != other.face:
            return FACES.index(self.face) <= FACES.index(other.face)
        else:
            return SUITS.index(self.suit) >= SUITS.index(other.suit)

    def __gt__(self, other):
        if self.face != other.face:
            return FACES.index(self.face) > FACES.index(other.face)
        else:
            return SUITS.index(self.suit) < SUITS.index(other.suit)

    def __ge__(self, other):
        if self.face != other.face:
            return FACES.index(self.face) >= FACES.index(other.face)
        else:
            return SUITS.index(self.suit) <= SUITS.index(other.suit)

    def __eq__(self, other):
        return self.face == other.face and self.suit == other.suit

    def __ne__(self, other):
        return self.face != other.face or self.suit != other.suit
