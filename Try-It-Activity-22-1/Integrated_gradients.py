# Integrated Gradients (IG) is a technique used to explain the predictions of machine learning models, particularly deep neural networks.
# It provides insights into which features of the input data contribute most to the model's predictions. This is especially useful in 
# computer vision tasks, where understanding how a model interprets images can help identify biases or errors in the model's decision-making process.
# This code demonstrates the Integrated Gradients technique using a pre-trained Inception V1 model from TensorFlow Hub. 
# The model is used to classify images, and the Integrated Gradients method is applied to visualize the contributions of different pixels 
# in the images to the model's predictions.
# The code includes loading the model, reading images, making predictions, and calculating Integrated Gradients. It also visualizes 
# the results, showing how different parts of the images contribute to the model's predictions. The final output includes attribution maps 
# that highlight the important features in the images for the model's predictions.
# The code is structured to be run in a Jupyter notebook or similar environment, where it can display images and plots inline. It uses TensorFlow 
# and TensorFlow Hub for model handling, and Matplotlib for visualization. The code is designed to be educational, providing a step-by-step 
# demonstration of the Integrated Gradients technique and its application to image classification tasks.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
import matplotlib.pylab as plt # type: ignore
import numpy as np             # type: ignore
import tensorflow as tf        # type: ignore
import tensorflow_hub as hub   # type: ignore
import logging                 # type: ignore
logging.getLogger('tensorflow').disabled       = True
logging.getLogger('tensorflow_hub').disabled   = True
logging.getLogger('tensorflow-macos').disabled = True
logging.getLogger('tensorflow-metal').disabled = True
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)  
warnings.filterwarnings('ignore', category=PendingDeprecationWarning)
warnings.filterwarnings('ignore', category=ImportWarning)
warnings.filterwarnings('ignore', category=SyntaxWarning)
warnings.filterwarnings('ignore', category=UnicodeWarning)
warnings.filterwarnings('ignore')
# Set matplotlib to use the 'Agg' backend for non-interactive plotting
plt.switch_backend('agg')
# This line of code loads a pre-trained Inception V1 neural network model from TensorFlow Hub, which serves as the foundation for demonstrating
# Integrated Gradients explainability techniques. TensorFlow Hub is Google's repository of pre-trained machine learning models that can be easily 
# integrated into new projects, eliminating the need to train complex models from scratch.
# The hub.KerasLayer() function creates a Keras layer wrapper around the pre-trained model, making it compatible with TensorFlow's high-level 
# Keras API. The URL "https://tfhub.dev/google/imagenet/inception_v1/classification/4" points to a specific version of the Inception V1 
# architecture that was trained on the ImageNet dataset. ImageNet is a massive dataset containing over 1 million images across 1,000 different 
# object categories, making this model capable of recognizing a wide variety of real-world objects from photographs.
# The input_shape=(224, 224, 3) parameter specifies the expected input dimensions for images fed into this model. The tuple represents 
# height (224 pixels), width (224 pixels), and color channels (3 for RGB). This standardized input size is crucial because neural networks 
# require consistent input dimensions, and 224×224 has become a common standard in computer vision due to its balance between detail preservation
# and computational efficiency. Images of different sizes must be resized to match this specification before being processed by the model.
# Setting trainable=False is particularly important in this context because it freezes the model's weights, preventing them from being updated 
# during any potential training operations. Since this is a pre-trained model being used for explainability analysis rather than further training, 
# keeping the weights frozen ensures that the model's learned representations remain consistent throughout the Integrated Gradients calculations. 
# This stability is essential for generating reliable attribution maps that accurately reflect how the original trained model makes its decisions.
# This choice of Inception V1 is especially relevant to the tutorial because it was one of the models featured in the original Integrated Gradients 
# research paper. Using the same architecture helps validate the explainability technique against established benchmarks while providing users with 
# a well-understood baseline for interpreting the attribution results.
model = hub.KerasLayer("https://tfhub.dev/google/imagenet/inception_v1/classification/4", input_shape=(224, 224, 3), trainable=False)
#
def load_imagenet_labels(file_path):
  labels_file = tf.keras.utils.get_file('ImageNetLabels.txt', file_path)
  with open(labels_file) as reader:
    f      = reader.read()
    labels = f.splitlines()
  return np.array(labels)
#
imagenet_labels = load_imagenet_labels('https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
#
def read_image(file_name):
  image = tf.io.read_file(file_name)
  image = tf.io.decode_jpeg(image, channels=3)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.image.resize_with_pad(image, target_height=224, target_width=224)
  return image
#
img_url = {
    'Fireboat'   : 'http://storage.googleapis.com/download.tensorflow.org/example_images/San_Francisco_fireboat_showing_off.jpg',
    'Giant Panda': 'http://storage.googleapis.com/download.tensorflow.org/example_images/Giant_Panda_2.jpeg',
    'coyote'     : 'https://upload.wikimedia.org/wikipedia/commons/c/ce/Canis_latrans_%28Yosemite%2C_2009%29.jpg'
}
img_paths        = {name: tf.keras.utils.get_file(f"{name}.jpg", url) for (name, url) in img_url.items()}
img_name_tensors = {name: read_image(img_path) for (name, img_path) in img_paths.items()}
#
plt.figure(figsize=(8, 8))
for n, (name, img_tensors) in enumerate(img_name_tensors.items()):
  ax = plt.subplot(1, 3, n+1)
  ax.imshow(img_tensors)
  ax.set_title(name)
  ax.axis('off')
plt.tight_layout()
plt.savefig('images/IG-01.jpg', dpi=300)
#
def top_k_predictions(img, k=3):
  image_batch         = tf.expand_dims(img, 0)
  predictions         = model(image_batch)
  probs               = tf.nn.softmax(predictions, axis=-1)
  top_probs, top_idxs = tf.math.top_k(input=probs, k=k)
  top_labels          = imagenet_labels[tuple(top_idxs)]
  return top_labels, top_probs[0]
#
print('----------------------------------')
for (name, img_tensor) in img_name_tensors.items():
  plt.imshow(img_tensor)
  plt.title(name, fontweight='bold')
  plt.axis('off')
  plt.savefig(f'images/IG-02-{name}.jpg', dpi=300)
  #plt.show()
  print(f'Top {len(imagenet_labels)} predictions for {name}:')
  print('----------------------------------')
  print(f'Image shape       : {img_tensor.shape}')
  print(f'Image dtype       : {img_tensor.dtype}')
  print(f'Image pixel range : {tf.reduce_min(img_tensor):0.2f} to {tf.reduce_max(img_tensor):0.2f}')
  print(f'Image pixel mean  : {tf.reduce_mean(img_tensor):0.2f}')
  print(f'Image pixel stddev: {tf.math.reduce_std(img_tensor):0.2f}')
  print('----------------------------------')
  print('Top predictions:')
  print('----------------------------------')
  pred_label, pred_prob = top_k_predictions(img_tensor)
  for label, prob in zip(pred_label, pred_prob):
    print(f'{label}: {prob:0.1%}')
#
def f(x):
  """A simplified model function."""
  return tf.where(x < 0.8, x, 0.8)
def interpolated_path(x):
  """A straight line path."""
  return tf.zeros_like(x)
x = tf.linspace(start=0.0, stop=1.0, num=6)
y = f(x)
#
fig = plt.figure(figsize=(12, 5))
ax0 = fig.add_subplot(121)
ax0.plot(x, f(x), marker='o')
ax0.set_title('Gradients saturate over F(x)', fontweight='bold')
ax0.text(0.2, 0.5 , 'Gradients > 0 = \n x is important')
ax0.text(0.7, 0.85, 'Gradients = 0 \n x not important')
ax0.set_yticks(tf.range(0, 1.5, 0.5))
ax0.set_xticks(tf.range(0, 1.5, 0.5))
ax0.set_ylabel('F(x) - model true class predicted probability')
ax0.set_xlabel('x - (pixel value)')
ax1 = fig.add_subplot(122)
ax1.plot(x, f(x), marker='o')
ax1.plot(x, interpolated_path(x), marker='>')
ax1.set_title('IG intuition', fontweight='bold')
ax1.text(0.25, 0.1, 'Accumulate gradients along path')
ax1.set_ylabel('F(x) - model true class predicted probability')
ax1.set_xlabel('x - (pixel value)')
ax1.set_yticks(tf.range(0, 1.5, 0.5))
ax1.set_xticks(tf.range(0, 1.5, 0.5))
ax1.annotate('Baseline', xy=(0.0, 0.0), xytext=(0.0, 0.2),
             arrowprops=dict(facecolor='black', shrink=0.1))
ax1.annotate('Input', xy=(1.0, 0.0), xytext=(0.95, 0.2),
             arrowprops=dict(facecolor='black', shrink=0.1))
plt.savefig('images/IG-03.jpg', dpi=300)
#plt.show()
#
baseline = tf.zeros(shape=(224,224,3))
plt.figure(figsize=(5, 5))
plt.imshow(baseline)
plt.title("Baseline")
plt.axis('off')
plt.savefig('images/IG-04.jpg', dpi=300)
#plt.show()
#
m_steps=50
alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)
#
def interpolate_images(baseline, image, alphas):
  alphas_x   = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
  baseline_x = tf.expand_dims(baseline, axis=0)
  input_x    = tf.expand_dims(image, axis=0)
  delta      = input_x - baseline_x
  images     = baseline_x +  alphas_x * delta
  return images
#
interpolated_images = interpolate_images(baseline=baseline,image=img_name_tensors['Fireboat'],alphas=alphas)
fig = plt.figure(figsize=(20, 20))
i = 0
for alpha, image in zip(alphas[0::10], interpolated_images[0::10]):
  i += 1
  plt.subplot(1, len(alphas[0::10]), i)
  plt.title(f'alpha: {alpha:.1f}')
  plt.imshow(image)
  plt.axis('off')
plt.tight_layout()
plt.savefig('images/IG-05.jpg', dpi=300)
#plt.show()
#
def compute_gradients(images, target_class_idx):
  with tf.GradientTape() as tape:
    tape.watch(images)
    logits = model(images)
    probs  = tf.nn.softmax(logits, axis=-1)[:, target_class_idx]
  return tape.gradient(probs, images)
#
path_gradients = compute_gradients(images = interpolated_images, target_class_idx = 555)
#
pred       = model(interpolated_images)
pred_proba = tf.nn.softmax(pred, axis=-1)[:, 555]
#
plt.figure(figsize=(10, 4))
ax1 = plt.subplot(1, 2, 1)
ax1.plot(alphas, pred_proba)
ax1.set_title('Target class predicted probability over alpha')
ax1.set_ylabel('model p(target class)')
ax1.set_xlabel('alpha')
ax1.set_ylim([0, 1])
ax2 = plt.subplot(1, 2, 2)
#
average_grads      = tf.reduce_mean(path_gradients, axis=[1, 2, 3])
average_grads_norm = (average_grads-tf.math.reduce_min(average_grads))/(tf.math.reduce_max(average_grads)-tf.reduce_min(average_grads))
# 
ax2.plot(alphas, average_grads_norm)
ax2.set_title('Average pixel gradients (normalized) over alpha')
ax2.set_ylabel('Average pixel gradients')
ax2.set_xlabel('alpha')
ax2.set_ylim([0, 1])
plt.tight_layout()
plt.savefig('images/IG-06.jpg', dpi=300)
#plt.show()
#
def integral_approximation(gradients):
  grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
  integrated_gradients = tf.math.reduce_mean(grads, axis=0)
  return integrated_gradients
#
ig = integral_approximation(gradients=path_gradients)
#
def integrated_gradients(baseline,image,target_class_idx,m_steps=50, batch_size=32):
  alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1) 
  gradient_batches = []
  for alpha in tf.range(0, len(alphas), batch_size):
    from_          = alpha
    to             = tf.minimum(from_ + batch_size, len(alphas))
    alpha_batch    = alphas[from_:to]
    gradient_batch = one_batch(baseline, image, alpha_batch, target_class_idx)
    gradient_batches.append(gradient_batch)
  total_gradients      = tf.concat(gradient_batches, axis=0)
  avg_gradients        = integral_approximation(gradients=total_gradients)
  integrated_gradients = (image - baseline) * avg_gradients
  return integrated_gradients
#
def one_batch(baseline, image, alpha_batch, target_class_idx):
    interpolated_path_input_batch = interpolate_images(baseline=baseline, image=image, alphas=alpha_batch)
    gradient_batch                = compute_gradients(images = interpolated_path_input_batch,target_class_idx=target_class_idx)
    return gradient_batch
#
ig_attributions = integrated_gradients(baseline=baseline, image=img_name_tensors['Fireboat'], target_class_idx=555, m_steps=240)
print('----------------------------------')
print('Attributions shape:', ig_attributions.shape)
print('----------------------------------')
#
def plot_img_attributions(name, baseline, image, target_class_idx, m_steps=50, cmap=None, overlay_alpha=0.4):
  attributions     = integrated_gradients(baseline=baseline, image=image, target_class_idx=target_class_idx, m_steps=m_steps)
  attribution_mask = tf.reduce_sum(tf.math.abs(attributions), axis=-1)
  fig, axs         = plt.subplots(nrows=2, ncols=2, squeeze=False, figsize=(8, 8))
  axs[0, 0].set_title('Baseline image')
  axs[0, 0].imshow(baseline)
  axs[0, 0].axis('off')
  axs[0, 1].set_title('Original image')
  axs[0, 1].imshow(image)
  axs[0, 1].axis('off')
  axs[1, 0].set_title('Attribution mask')
  axs[1, 0].imshow(attribution_mask, cmap=cmap)
  axs[1, 0].axis('off')
  axs[1, 1].set_title('Overlay')
  axs[1, 1].imshow(attribution_mask, cmap=cmap)
  axs[1, 1].imshow(image, alpha=overlay_alpha)
  axs[1, 1].axis('off')
  plt.tight_layout()
  plt.savefig('images/IG-07-'+name+'.jpg', dpi=300)
  #plt.show()
  return fig
_ = plot_img_attributions(name='Fireboat',    image=img_name_tensors['Fireboat'],   baseline=baseline, target_class_idx=555, m_steps=240, cmap=plt.cm.inferno, overlay_alpha=0.4)
_ = plot_img_attributions(name='Giant Panda', image=img_name_tensors['Giant Panda'],baseline=baseline, target_class_idx=389, m_steps=55,  cmap=plt.cm.viridis, overlay_alpha=0.5)
_ = plot_img_attributions(name='coyote',      image=img_name_tensors['coyote']     ,baseline=baseline, target_class_idx=389, m_steps=55,  cmap=plt.cm.viridis, overlay_alpha=0.5)