from src.core.preprocessing import load_img
from src.core.lbp import compute_lbp_classification
import numpy as np
import cv2


def main():
    file = "mandrill"

    rgb_img = load_img(f"./data/png/{file}.png")
    classification_map: np.ndarray = compute_lbp_classification(rgb_img)
    bw_img = classification_map * 255

    # cv2.imwrite(f"./data/lbp/{file}_3msb.png", bw_img)
    cv2.imwrite(f"./data/lbp/{file}_8-bit.png", bw_img)


if __name__ == "__main__":
    main()
