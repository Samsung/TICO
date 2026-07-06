# TICO Logo

## Concept

The TICO symbol is an **open circle** — the letter **C** for *Circle*, drawn as a bold
ring with one segment missing, and a **blue dot** arriving to close it.

It tells the story of what TICO does in a single gesture: a PyTorch model — the dot —
travels through the conversion pipeline and lands exactly where it completes the
Circle. Conversion is the moment the circle becomes whole.

- **The ring** — the Circle format, ONE's lightweight on-device representation.
  Drawn with a heavy, rounded stroke: stable, efficient, production-ready.
- **The dot** — the incoming Torch IR module, rendered in blue as the single
  point of energy in an otherwise monochrome mark. Its placement on the ring's
  open edge represents `tico.convert()`: one call, and the model snaps into place.
- **The opening** — TICO itself: the gap between PyTorch and on-device inference
  that this library closes.

## Files

| Path | Usage |
|---|---|
| `banner/tico-banner-light.png` | README hero banner, light mode (1280×320) |
| `banner/tico-banner-dark.png` | README hero banner, dark mode (1280×320) |
| `favicon/tico-{16..512}.png` | Favicons / app icons, black symbol on transparent |
| `favicon/tico-512-white.png` | White symbol on transparent, for dark surfaces |
| `profile/tico-profile-512.png` | GitHub organization / repository avatar |
| `svg/tico-symbol.svg` | Vector master, black symbol |
| `svg/tico-symbol-white.svg` | Vector master, white symbol |
| `svg/tico-profile.svg` | Vector master, avatar (white on black disc) |

## Colors & type

- Symbol: black `#000000` (light) / white `#FFFFFF` (dark)
- Accent dot: blue `#0066FF` — the only color in the mark
- Wordmark: Manrope ExtraBold (800), tight letter-spacing

## Usage notes

- Prefer the SVG masters; regenerate raster sizes from them.
- Keep the dot blue — it carries the meaning of the mark.
- Leave clear space around the symbol of at least the dot's diameter.
- Do not rotate the mark: the opening faces right, where the dot lands.
