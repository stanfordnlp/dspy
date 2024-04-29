import random

from .dataset import Dataset
from dspy.primitives.example import Example
from pydantic import Field

### A bunch of colors, originally from matplotlib
_all_colors = [
    "alice blue",
    "dodger blue",
    "light sky blue",
    "deep sky blue",
    "sky blue",
    "steel blue",
    "light steel blue",
    "medium blue",
    "navy blue",
    "blue",
    "royal blue",
    "cadet blue",
    "cornflower blue",
    "medium slate blue",
    "slate blue",
    "dark slate blue",
    "powder blue",
    "turquoise",
    "dark turquoise",
    "medium turquoise",
    "pale turquoise",
    "light sea green",
    "medium sea green",
    "sea green",
    "forest green",
    "green yellow",
    "lime green",
    "dark green",
    "green",
    "lime",
    "chartreuse",
    "lawn green",
    "yellow green",
    "olive green",
    "dark olive green",
    "medium spring green",
    "spring green",
    "medium aquamarine",
    "aquamarine",
    "aqua",
    "cyan",
    "dark cyan",
    "teal",
    "medium orchid",
    "dark orchid",
    "orchid",
    "blue violet",
    "violet",
    "dark violet",
    "plum",
    "thistle",
    "magenta",
    "fuchsia",
    "dark magenta",
    "medium purple",
    "purple",
    "rebecca purple",
    "dark red",
    "fire brick",
    "indian red",
    "light coral",
    "dark salmon",
    "light salmon",
    "salmon",
    "red",
    "crimson",
    "tomato",
    "coral",
    "orange red",
    "dark orange",
    "orange",
    "yellow",
    "gold",
    "light goldenrod yellow",
    "pale goldenrod",
    "goldenrod",
    "dark goldenrod",
    "beige",
    "moccasin",
    "blanched almond",
    "navajo white",
    "antique white",
    "bisque",
    "burlywood",
    "dark khaki",
    "khaki",
    "tan",
    "wheat",
    "snow",
    "floral white",
    "old lace",
    "ivory",
    "linen",
    "seashell",
    "honeydew",
    "mint cream",
    "azure",
    "lavender",
    "ghost white",
    "white smoke",
    "gainsboro",
    "light gray",
    "silver",
    "dark gray",
    "gray",
    "dim gray",
    "slate gray",
    "light slate gray",
    "dark slate gray",
    "black",
    "medium violet red",
    "pale violet red",
    "deep pink",
    "hot pink",
    "light pink",
    "pink",
    "peach puff",
    "rosy brown",
    "saddle brown",
    "sandy brown",
    "chocolate",
    "peru",
    "sienna",
    "brown",
    "maroon",
    "white",
    "misty rose",
    "lavender blush",
    "papaya whip",
    "lemon chiffon",
    "light yellow",
    "corn silk",
    "pale green",
    "light green",
    "olive drab",
    "olive",
    "dark sea green",
]


class Colors(Dataset):
    sort_by_suffix: bool = Field(default=True)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.load()

    def load(self) -> None:
        if self.sort_by_suffix:
            colors = self.sort_by_suffix(_all_colors)
        else:
            colors = _all_colors

        train_size = int(len(colors) * 0.6)
        self.data["train"] = [Example(color=color) for color in colors[:train_size]]
        self.data["dev"] = [Example(color=color) for color in colors[train_size:]]

    def sorted_by_suffix(self, colors):
        if not self.sort_by_suffix:
            return colors

        if isinstance(colors[0], str):
            sorted_colors = sorted(colors, key=lambda x: x[::-1])
        else:
            sorted_colors = sorted(colors, key=lambda x: x["color"][::-1])

        return sorted_colors
