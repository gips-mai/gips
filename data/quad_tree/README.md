## Quad Tree

- In the OSV5M paper, the authors test multiple methods to predict the latitude and longitude geolocation of an image.
- Their best performing method is a hybrid approach which combines classification and regression:
  - For this, the authors divide the world into a grid of cells for which they classify the likelihood of the image being in each cell.
  - They then regress the latitude and longitude within the cell with the highest classification score.