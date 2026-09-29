# pt2_to_circle_test

`pt2_to_circle_test` validates the circle model which generated from pt2 file.

The test proceeds as follows

1. Generate `pt2` model from `torch` model and export it to `.pt2` format.
2. Load `pt2` model from the exported file, and convert it to the `circle` model using `pt2-to-circle`.
3. Validate converted `circle` model with `tico.circle` (structural verification, operator support, and shape/dtype contracts through the reference runtime).
4. Execute and compare the results from `torch` model and `circle` model. The results must be the same.
