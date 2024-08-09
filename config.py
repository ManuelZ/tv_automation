APPS = {
    0: "Youtube",
    1: "Television",
    2: "Netflix",
    3: "Max",
    4: "Internet",
    5: "Prime video",
    6: "TV en vivo",
    7: "Movistar TV App",
    8: "Spotify",
}
NUM_APPS = len(APPS.values())
WORLD = ["Configuration", "Origin", "Search", "Apps"] + list(APPS.values())
TARGET_W = 256
TARGET_H = 256
TARGET_SIZE = (TARGET_H, TARGET_W)
MARGIN_X = 1
DEBUG = True

# For blur detection, higher is better
MIN_VARIANCE_OF_LAPLACIAN = 35
