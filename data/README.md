# Data

The study uses the public Kaggle dataset **Chess Game Dataset (Lichess)**: https://www.kaggle.com/datasets/datasnaek/chess

It is **not redistributed** here. Check the licence on the dataset page, download `games.csv` and place it at `data/games.csv`.

It contains 20,058 games (August 2013 to September 2017) with both players' Lichess ids. Those ids are public Lichess usernames, which is another reason the file is not copied into this repository. Results only report aggregates per segment.

## Columns used

| column | meaning |
|---|---|
| `white_id`, `black_id` | Lichess usernames |
| `white_rating`, `black_rating` | rating of each side in that game |
| `winner` | `white`, `black` or `draw` |
| `victory_status` | `resign`, `mate`, `outoftime` or `draw` |
| `turns` | number of plies (half-moves); equals the length of the `moves` string |
| `increment_code` | time control `minutes+increment` |
| `opening_name` | Lichess opening name |
| `rated`, `created_at` | used only in the data audit |

`opening_ply` is deliberately **not** used: for 96.6% of opening names it takes a single value, so it describes the opening, not the player.
