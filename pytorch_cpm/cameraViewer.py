import time
import os
import sys
import cv2


def guessSource(source=None):
    import glob
    from pathlib import Path
    img_formats = ('bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo')  # acceptable image suffixes
    vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
    img_paths = []
    video_id = None
    if source is None:
        video_id = 0
    else:
        p = str(Path(source).absolute())  # os-agnostic absolute path
        if p.endswith('.txt') and os.path.isfile(p):
            with open(p, 'r') as f:
                files = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        elif '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        elif p.isdigit():
            video_id = int(p)
            return img_paths, video_id
        else:
            raise Exception(f'ERROR: {p} does not exist')

        print("files", files)
        img_paths = [x for x in files if x.split('.')[-1].lower() in img_formats]
        vid_paths = [x for x in files if x.split('.')[-1].lower() in vid_formats]
        if vid_paths:
            video_id = vid_paths[0]

    return img_paths, video_id


def generate_camera(cap):
    while True:
        ret, img = cap.read()
        if not ret:
            break
        # print(img.shape)
        yield img


def genImgs(img_paths):
    for i in img_paths:
        img = cv2.imread(i, 1)
        if img is None:
            print(i, "not found")
        else:
            yield img


class CameraViewer:
    def __init__(self, source, args=None):
        self.source = source
        self.args = args

        img_paths, video_id = guessSource(source)
        if img_paths:
            self.imgs = genImgs(img_paths)
            self.use_camera = False
            print("use images")
            self.title = "img_{i}"
        else:
            cap = cv2.VideoCapture(video_id)
            assert cap.isOpened(), f'Failed to open {video_id}'
            self.imgs = generate_camera(cap)
            self.use_camera = True
            print("use camera")
            self.title = "camera"

        self.waitTime = 30 if self.use_camera else 0

    def stream(self):
        try:
            yield from self.imgs
        except GeneratorExit as e:
            print("GeneratorExit", e)
        finally:
            # if self.use_camera:
            #     self.cap.release()
            cv2.destroyAllWindows()


def imageViewWrap(source, args=None):
    W, H = 256, 256
    cmv = CameraViewer(source, args)
    imgs = cmv.stream()
    for i, img in enumerate(imgs):
        t0 = time.time()
        # print("time: ", time.time() - t0)
        cv2.imshow(cmv.title.format(i), img)
        k = cv2.waitKey(cmv.waitTime)
        if k == 27:
            break
    print("before")

def predictWrap(source, model, args=None):
    H, W = model.height, model.width
    cmv = CameraViewer(source, args)
    imgs = cmv.stream()
    for i, img in enumerate(imgs):
        t0 = time.time()
        if img.shape[0:2] != (H, W):
            img = cv2.resize(img, (H, W)) 
        kpts = model.predict(img)

        print("time: ", time.time() - t0)
        img2 = model.draw(img, kpts)
        # cv2.imwrite('test.jpg', img2)
        cv2.imshow(cmv.title.format(i=i), img2)
        k = cv2.waitKey(cmv.waitTime)
        if k == 27:
            break
    print("before")


def parse_args(cmds=None):
    import argparse
    parser = argparse.ArgumentParser(description='Train Face Alignment')
    parser.add_argument('--input', '-i', help='input source')

    args = parser.parse_args(cmds)
    return args

def main(cmds=None):
    args = parse_args(cmds)
    imageViewWrap(args.input)

if __name__ == "__main__":
    cmds = ["-i", r"H:\Dataset\mov\湘AE0959.mp4"]
    main()
    