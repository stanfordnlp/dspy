import random

from dspy.datasets.dataset import Dataset

### A bunch of colors, originally from matplotlib
all_colors = [
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
    """Toy color-name dataset used in DSPy examples and tutorials.

    Wraps the matplotlib-derived ``all_colors`` list and splits it into
    train/dev partitions. By default, names are sorted by their reversed
    string so that visually similar colors (for example, the various ``blue``
    suffixes) cluster together; the first 60% becomes the train split and the
    rest becomes dev. Both splits are then deterministically shuffled with
    seed ``0`` before being handed to the ``Dataset`` base class for further
    seeded sampling.

    Args:
        sort_by_suffix: If ``True`` (the default), sort colors by reversed
            name so that similar colors are not split between train and dev.
            If ``False``, preserve the original ``all_colors`` ordering.
        *args: Forwarded to ``Dataset.__init__``.
        **kwargs: Forwarded to ``Dataset.__init__``.

    Examples:
        >>> from dspy.datasets.colors import Colors
        >>> data = Colors(input_keys=["color"])
        >>> example = data.train[0]
        >>> "color" in example
        True
    """

    def __init__(self, sort_by_suffix=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.sort_by_suffix = sort_by_suffix
        colors = self.sorted_by_suffix(all_colors)

        train_size = int(
            len(colors) * 0.6
        )  # chosen to ensure that similar colors aren't repeated between train and dev
        train_colors, dev_colors = colors[:train_size], colors[train_size:]

        self._train = [{"color": color} for color in train_colors]
        self._dev = [{"color": color} for color in dev_colors]

        random.Random(0).shuffle(self._train)
        random.Random(0).shuffle(self._dev)

    def sorted_by_suffix(self, colors):
        """Return ``colors`` sorted by reversed name when ``sort_by_suffix`` is set.

        Sorting on the reversed string groups colors with the same suffix
        (for example, ``"navy blue"`` and ``"sky blue"``) so a contiguous
        train/dev split keeps related colors together rather than scattering
        them across both partitions.

        Args:
            colors: Either a list of color-name strings, or a list of dicts
                with a ``"color"`` key.

        Returns:
            The input list sorted by reversed color name when
            ``self.sort_by_suffix`` is ``True``, otherwise the input
            unchanged.
        """
        if not self.sort_by_suffix:
            return colors

        if isinstance(colors[0], str):
            sorted_colors = sorted(colors, key=lambda x: x[::-1])
        else:
            sorted_colors = sorted(colors, key=lambda x: x["color"][::-1])

        return sorted_colors
