import matplotlib.pyplot as plt
from imageio import imread
from alpha_blend import alpha_blend

if __name__ == "__main__":
    # Load input image, compute blended image.
    Ifg = imread('../images/stripes.png')
    Ibg = imread('../images/earth.png')
    Ia = alpha_blend(Ifg, Ibg, 0.3)

    plt.imshow(Ia)
    plt.show()