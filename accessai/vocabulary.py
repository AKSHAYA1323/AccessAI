"""Maps everyday words to the COCO classes the detector is able to confirm."""

import re

COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
)

_PEOPLE = (
    "man", "woman", "boy", "girl", "child", "kid", "baby", "people", "guy", "lady", "gentleman",
    "toddler", "player", "skier", "surfer", "snowboarder", "skateboarder", "rider", "pedestrian",
    "human", "adult", "teenager", "bride", "groom", "chef", "worker", "student",
)

_SYNONYMS = {
    "bike": ("bicycle", "motorcycle"), "motorbike": ("motorcycle",), "scooter": ("motorcycle",),
    "plane": ("airplane",), "aeroplane": ("airplane",), "jet": ("airplane",),
    "automobile": ("car",), "taxi": ("car",), "cab": ("car",), "van": ("car", "truck"),
    "ship": ("boat",), "sailboat": ("boat",), "hydrant": ("fire hydrant",),
    "puppy": ("dog",), "kitten": ("cat",), "calf": ("cow",), "lamb": ("sheep",),
    "bag": ("handbag", "backpack", "suitcase"), "purse": ("handbag",), "luggage": ("suitcase",),
    "rucksack": ("backpack",), "necktie": ("tie",), "ball": ("sports ball",),
    "soccer ball": ("sports ball",), "tennis ball": ("sports ball",),
    "racket": ("tennis racket",), "racquet": ("tennis racket",), "bat": ("baseball bat",),
    "glove": ("baseball glove",), "ski": ("skis",), "glass": ("wine glass", "cup"),
    "mug": ("cup",), "coffee": ("cup",), "doughnut": ("donut",), "hotdog": ("hot dog",),
    "sofa": ("couch",), "plant": ("potted plant",), "table": ("dining table",),
    "desk": ("dining table",), "stool": ("chair",), "television": ("tv",), "monitor": ("tv",),
    "screen": ("tv", "laptop"), "computer": ("laptop", "tv"), "notebook": ("laptop", "book"),
    "phone": ("cell phone",), "cellphone": ("cell phone",), "smartphone": ("cell phone",),
    "fridge": ("refrigerator",), "teddy": ("teddy bear",), "hairdryer": ("hair drier",),
    "toilet seat": ("toilet",), "sign": ("stop sign",), "earbuds": ("earbud",),
    "earpods": ("earbud",), "airpod": ("earbud",), "airpods": ("earbud",),
    "earphone": ("earbud",), "earphones": ("earbud",), "pen": ("pen",),
}

# Word (lower case, singular) -> COCO classes that could confirm it.
TERM_TO_CLASSES: dict[str, tuple[str, ...]] = {
    **{name: (name,) for name in COCO_CLASSES},
    **{word: ("person",) for word in _PEOPLE},
    **_SYNONYMS,
}

# Longest first, so that "traffic light" is matched before "light".
TERMS_LONGEST_FIRST = sorted(TERM_TO_CLASSES, key=len, reverse=True)

# Words about the picture itself, or quantities rather than things. Never checked as claims.
META_NOUNS = {
    "image", "picture", "photo", "photograph", "view", "scene", "shot", "close", "closeup",
    "group", "bunch", "couple", "pair", "lot", "number", "piece", "variety", "row", "side",
    "front", "back", "top", "bottom", "middle", "background", "foreground", "center", "thing",
    "something", "someone", "somebody", "there", "it", "that", "this",
}

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10,
}


def find_term(text: str) -> str | None:
    """Return the longest vocabulary term contained in text as whole words, if any."""
    padded = f" {' '.join(re.findall(r'[a-z0-9]+', text.lower()))} "
    for term in TERMS_LONGEST_FIRST:
        for form in (term, _plural(term)):
            if f" {form} " in padded:
                return term
    return None


def _plural(term: str) -> str:
    irregular = {
        "man": "men", "woman": "women", "child": "children", "person": "people",
        "knife": "knives", "calf": "calves", "bus": "buses", "glass": "glasses",
    }
    if term in irregular:
        return irregular[term]
    return term + ("es" if term.endswith(("s", "x", "ch", "sh")) else "s")
