import io
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf


default_color = 'blue'
highlight_color = 'red'


class DetectionOverlay:
  
  def __init__(self, args):
    self.args = args
    self.labels_to_highlight = args.labels_to_highlight.split(";")
    self.font = ImageFont.load_default()

  def apply_overlay(self, sess, image_bytes, feature, img_width, img_height):
    """Apply annotation overlay over input image.
    
    Args:
      image_bytes: JPEG image.
      feature: TF Record Feature

    Returns:
      image_bytes_with_overlay: JPEG image with annotation overlay.
    """

    bboxes = self.get_bbox_tuples(sess, feature)
    image_bytes_with_overlay = self.draw_bboxes(image_bytes, bboxes, img_width, img_height)
    return image_bytes_with_overlay


  def get_bbox_tuples(self, sess, feature):
    """ From a TF Record Feature, get a list of tuples representing bounding boxes
    
    Args:
      feature: TF Record Feature
    Returns:
      bboxes (list of tuples): [ (label, xmin, xmax, ymin, ymax), (label, xmin, xmax, ymin, ymax) , .. ]
    """
    bboxes = []
    if self.args.bbox_name_key in feature:
      gtboxes_and_label = tf.decode_raw(feature[self.args.bbox_name_key].bytes_list.value[0], tf.int32)
      with sess.as_default():
        gtboxes_and_label = tf.reshape(gtboxes_and_label, [-1, 5]).eval()
      for bbox in gtboxes_and_label:
        bboxes.append((
          bbox[0],  # xmin
          bbox[1],  # ymin
          bbox[2],  # xmax
          bbox[3],  # ymax
          bbox[4],  # label
        ))
    else:
      print("Bounding box key '%s' not present." % (self.args.bbox_name_key))
    return bboxes

  def bbox_color(self, label):
    if label in self.labels_to_highlight:
      return highlight_color
    else:
      return default_color

  def bboxes_to_pixels(self, bbox, im_width, im_height):
    """
    Convert bounding box coordinates to pixels.
    (It is common that bboxes are parametrized as percentage of image size
    instead of pixels.)

    Args:
      bboxes (tuple): (label, xmin, xmax, ymin, ymax)
      im_width (int): image width in pixels
      im_height (int): image height in pixels
    
    Returns:
      bboxes (tuple): (label, xmin, xmax, ymin, ymax)
    """
    if self.args.coordinates_in_pixels:
      return bbox
    else:
      xmin, ymin, xmax, ymax, label = bbox
      return [xmin * im_width, ymin * im_height, xmax * im_width, ymax * im_height, label]

  def draw_bboxes(self, image_bytes, bboxes, img_width, img_height):
    """Draw bounding boxes onto image.
    
    Args:
      image_bytes: JPEG image.
      bboxes (list of tuples): [ (label, xmin, xmax, ymin, ymax), (label, xmin, xmax, ymin, ymax) , .. ]
    
    Returns:
      image_bytes: JPEG image including bounding boxes.
    """
    img = Image.frombytes('RGB', (img_width, img_height), image_bytes, 'raw')
    # image.show()

    # img = Image.open(io.BytesIO(image_bytes))

    draw = ImageDraw.Draw(img)

    width, height = img.size

    for bbox in bboxes:
      # xmin, ymin, xmax, ymax, label = self.bboxes_to_pixels(bbox, width, height)
      xmin, ymin, xmax, ymax, label = bbox
      label = str(label)
      draw.rectangle([xmin, ymin, xmax, ymax], outline=self.bbox_color(label))

      w, h = self.font.getsize(label)
      draw.rectangle((xmin, ymin, xmin + w + 4, ymin + h), fill="white")

      draw.text((xmin+4, ymin), label, fill=self.bbox_color(label), font=self.font)

    with io.BytesIO() as output:
      if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
      img.save(output, format="JPEG")
      output_image = output.getvalue()
    return output_image
