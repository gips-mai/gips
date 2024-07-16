## Quad Tree

- In the OSV5M paper, the authors test multiple methods to predict the latitude and longitude geolocation of an image.
- Their best performing method is a hybrid approach which combines classification and regression:
  - For this, the authors divide the world into a grid of cells for which they classify the likelihood of the image being in each cell.
  - Based on the center coordinates within the cell with the highest classification score, the authors then regress the latitude and longitude.
  - For this approach the authors use a quad tree to divide the world into cells, which we copied from the OSV5M repository.
- The hybrid head is implemented in the [geolocation head class](..%2F..%2Fmodel%2Fmodules%2Fheads%2Fgeoloc_head.py)
