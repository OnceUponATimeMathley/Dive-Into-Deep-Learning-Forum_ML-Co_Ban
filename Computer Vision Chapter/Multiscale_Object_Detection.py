
from mxnet import image, np, npx
from d2l import mxnet as d2l

npx.set_np()

img = image.imread('img/catdog.jpg')
h, w = img.shape[0:2]


def multibox_prior(data, sizes, ratios):
    #data: batch, channels, height, width
    in_height, in_width = data.shape[-2:]

    device, num_sizes, num_ratios = data.ctx, len(sizes), len(ratios)
    boxes_per_pixel = num_sizes + num_ratios - 1
    size_tensor = np.array(sizes, ctx=device)
    ratio_tensor = np.array(ratios, ctx=device)

    # Offsets are required to move the anchor to center of a pixel
    # Since pixel (height=1, width=1), we choose to offset our centers by 0.5
    offset_w, offset_h = 0.5, 0.5
    steps_h = 1.0 / in_height # Scaled steps in y axis
    steps_w = 1.0 / in_width # Scaled steps in x axis

    # Generate all center points for the anchor boxes
    center_h = (np.arange(in_height, ctx=device) + offset_h) * steps_h
    center_w = (np.arange(in_width, ctx=device) + offset_w) * steps_w
    shift_x, shift_y = np.meshgrid(center_w, center_h)
    shift_x, shift_y = shift_x.reshape(-1), shift_y.reshape(-1)

    # Generate boxes_per_pixel number of heights and widths which are later
    # used to create anchor box corner coordinates (xmin, xmax, ymin, ymax)
    # concat (various sizes, first ratio) and (first size, various ratios)

    w = np.concatenate((size_tensor * np.sqrt(ratio_tensor[0]),
                        size_tensor[0]* np.sqrt(ratio_tensor[1:])))\
                        * in_height / in_width

    h = np.concatenate((size_tensor / np.sqrt(ratio_tensor[0]),
                        sizes[0] / np.sqrt(ratio_tensor[1:])))

    # Divide by 2 to get half height and half width
    anchor_manipulations = np.tile(np.stack((-w, -h, w, h)).T,
                                   (in_height * in_width, 1)) / 2

    # Each center point will have boxes_per_pixel number of anchor boxes, so
    # generate grid of all anchor box centers with boxes_per_pixel repeats
    out_grid = np.stack([shift_x, shift_y, shift_x, shift_y],
                        axis=1).repeat(boxes_per_pixel, axis=0)

    output = out_grid + anchor_manipulations
    # print(output)
    print(in_height, in_width)
    return np.expand_dims(output, axis=0)


def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    # The values from the first two dimensions will not affect the output
    fmap = np.zeros((1, 10, fmap_h, fmap_w))
    anchors = multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    print(anchors.shape)
    bbox_scale = np.array((w, h, w, h))
    d2l.show_bboxes(
        d2l.plt.imshow(img.asnumpy()).axes, anchors[0] * bbox_scale)

display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
# d2l.plt.show()