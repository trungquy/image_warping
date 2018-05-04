import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
from scipy.misc import imread, imsave


class Point(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return "{}-{}".format(self.x, self.y)


class Rect(object):

    def __init__(self, points, h, w):
        """
        Input:
            - h, w: height, width of image which rect belong to
        """
        assert len(points) == 4
        (self.left, self.right, self.top, self.bottom) = get_left_right_top_bottom(h, w, points)
        self.points = points

    def is_inside(self, point):
        return point.x >= self.left and point.x < self.right \
            and point.y >= self.top and point.y <= self.bottom

    def area(self):
        return (self.bottom - self.top + 1) * (self.right - self.left + 1)


class PairPoints(object):

    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2


def get_coresponding_pair_points(points, padding=0):
    # return [
    #     PairPoints(Point(34, 56), Point(34, 56)),
    #     PairPoints(Point(34, 56), Point(34, 56)),
    #     PairPoints(Point(34, 56), Point(34, 56)),
    #     PairPoints(Point(34, 56), Point(34, 56)),
    # ]
    assert len(points) == 4
    # h = points[2].x - points[1].x
    # w = points[1].y - points[0].y
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    top = int(min(ys))
    bottom = int(max(ys))

    right = int(max(xs))
    left = int(min(xs))
    # pad_x = 
    h = bottom - top
    w = right - left

    p2_points = [
        Point(0 + padding, 0 + padding),
        Point(w + padding, 0 + padding),
        Point(w + padding, h + padding),
        Point(0 + padding, h + padding)
    ]
    pair_points = []
    for p1, p2 in zip(points, p2_points):
        pair_points.append(PairPoints(p1, p2))
    return pair_points


def compute_homograpy_matrix(pair_points):
    """
    aX = b
    X is 3x3 matrix, X[3,3] = 1
    => Need to compute 8 values using linear least square method
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.lstsq.html
    """
    a = []
    b = []
    for pair_point in pair_points:
        p1 = pair_point.p1
        p2 = pair_point.p2
        a.append([p1.x, p1.y, 1, 0, 0, 0, -p1.x * p2.x, -p1.y * p2.x])
        a.append([0, 0, 0, p1.x, p1.y, 1, -p1.x * p2.y, -p1.y * p2.y])
        b.append(p2.x)
        b.append(p2.y)
    a = np.asarray(a)
    b = np.asarray(b)
    H_flatten = np.linalg.lstsq(a=a, b=b)[0]
    H_flatten = np.asarray(H_flatten.tolist() + [1])

    return np.reshape(H_flatten, (3, 3))


def rectify_image(im, H, cropped_rect=None, crop_output=True, padding=0):
    """
    rectify_image by Homograpy matrix
    """
    h, w = im.shape[:2]
    rectified_im = np.zeros(shape=(im.shape[0] + 2 * padding, im.shape[1] + 2 * padding, im.shape[2]))
    cnt = 0
    cnt_total = 0
    most_right, most_bottom = (0, 0)
    for i in range(w):
        for j in range(h):
            if cropped_rect and not cropped_rect.is_inside(Point(i, j)):
                continue
            cnt_total += 1
            p1 = np.asarray([[i], [j], [1.0]])
            p2 = np.matmul(H, p1).squeeze()
            # p2 = np.squeeze(p2)
            # print p2
            p2 = Point(int(p2[0] * 1.0 / p2[2]), int(p2[1] * 1.0 / p2[2]))
            if p2.x >= 0 and p2.y >= 0 and p2.y <= h and p2.x <= w:
                # print p2
                rectified_im[p2.y, p2.x, :] = im[j, i, :]
                if most_right < p2.x:
                    most_right = p2.x
                if most_bottom < p2.y:
                    most_bottom = p2.y
                cnt += 1
    # total = (h * w) if not cropped_rect else cropped_rect.area()
    # assert total == cnt_total, "{} != {}".format(total, cnt_total)
    print "cnt: {}/{} ({}%)".format(cnt, cnt_total, cnt * 100.0 / cnt_total)
    # print
    if not crop_output:
        return rectified_im
    else:
        return rectified_im[:most_bottom,:most_right,:]

def pick_4_points_on_the_same_plane(im_path):
    fig, ax = plt.subplots()
    ax.set_title("Pick 4 points - The same plane")
    im = imread(im_path)
    ax.imshow(im, picker=True)
    points = []

    def onpick_image(event):
        artist = event.artist
        print event
        if isinstance(event.artist, AxesImage):
            im_t = artist
            A = im_t.get_array()
            print("onpick_point", A.shape)

    def onclick_point(event):
        # print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        #       (event.button, event.x, event.y, event.xdata, event.ydata))
        # points.append(Point(int(event.ydata), int(event.xdata)))
        points.append(Point(int(event.xdata), int(event.ydata)))
        if len(points) >= 4:
            plt.close(fig)
            print "Points: {}".format(points)

    # fig.canvas.mpl_connect('pick_event', onpick_images)
    fig.canvas.mpl_connect('button_press_event', onclick_point)
    plt.show()
    return points


def get_left_right_top_bottom(h, w, points):
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    top = int(max(0, min(ys)))
    bottom = int(min(h, max(ys)))

    right = int(min(w, max(xs)))
    left = int(max(0, min(xs)))
    return (left, right, top, bottom)


def crop_image(im, points, padding=0):
    assert len(points) == 4
    h, w = im.shape[:2]
    (left, right, top, bottom) = get_left_right_top_bottom(h, w, points)
    print (left, right, top, bottom)
    if padding and padding > 0:
        padded_points = [
            Point(left - padding, top - padding),
            Point(right + padding, top - padding),
            Point(right + padding, bottom + padding),
            Point(left - padding, bottom + padding),
        ]
        cropped_rect = Rect(padded_points, h, w)
    else:
        cropped_rect = Rect(points,h, w)

    cropped_im = im[top - padding:bottom + padding, left - padding:right + padding, :]
    return cropped_im, cropped_rect


def main():
    im_path = "arsenal_wall.jpg"
    im = imread(im_path)
    h, w = im.shape[:2]
    padding = int(0.2 * min(w, h))
    # padding = 0
    p1_points = pick_4_points_on_the_same_plane(im_path)
    pair_points = get_coresponding_pair_points(points=p1_points, padding=padding)
    H = compute_homograpy_matrix(pair_points=pair_points)
    # cropped_im, crop_rect = crop_image(im, points=p1_points, padding=padding)
    cropped_im, crop_rect = crop_image(im, points=p1_points)
    print "Cropped Image : {}".format(cropped_im.shape)
    imsave("in.jpg", cropped_im)
    # rectified_im = rectify_image(im, H, crop_rect, padding=padding)
    rectified_im = rectify_image(im, H, padding=padding)
    print "Rectified Image : {}".format(rectified_im.shape)
    imsave("out.jpg", rectified_im)
    # plt.figure()
    # plt.imshow(rectified_im)
    # plt.show()

if __name__ == "__main__":
    main()
