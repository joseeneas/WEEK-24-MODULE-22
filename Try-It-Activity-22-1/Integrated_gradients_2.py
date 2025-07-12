#------------------------------------------------------------------------------------------------------------------------------------------
# Integrated Gradients (IG) is a technique used to explain the predictions of machine learning models, particularly deep neural networks.
# It provides insights into which features of the input data contribute most to the model's predictions. This is especially useful in 
# computer vision tasks, where understanding how a model interprets images can help identify biases or errors in the model's decision-making
# process. This code demonstrates the Integrated Gradients technique using a pre-trained Inception V1 model from TensorFlow Hub. 
# The model is used to classify images, and the Integrated Gradients method is applied to visualize the contributions of different pixels 
# in the images to the model's predictions.
# The code includes loading the model, reading images, making predictions, and calculating Integrated Gradients. It also visualizes 
# the results, showing how different parts of the images contribute to the model's predictions. The final output includes attribution maps 
# that highlight the important features in the images for the model's predictions.
# The code is structured to be run in a Jupyter notebook or similar environment, where it can display images and plots inline. 
# It uses TensorFlow # and TensorFlow Hub for model handling, and Matplotlib for visualization. The code is designed to be educational, 
# providing a step-by-step # demonstration of the Integrated Gradients technique and its application to image classification tasks.
#------------------------------------------------------------------------------------------------------------------------------------------
# This code is part of the "Try It" activity for the "Explainable AI with TensorFlow" course, specifically for the "Integrated Gradients" 
# section. It is designed to demonstrate the Integrated Gradients technique, which is a method for attributing the output of a machine
# learning model to its input features. The code uses a pre-trained Inception V1 model from TensorFlow Hub to classify images and then applies
# the Integrated Gradients method to visualize the contributions of different pixels in the images to the model's predictions.
# The code is structured to be run in a Jupyter notebook or similar environment,
# where it can display images and plots inline. It uses TensorFlow and TensorFlow Hub for model handling, and Matplotlib for visualization.
# The code is designed to be educational, providing a step-by-step demonstration of the Integrated Gradients technique and its application 
# to image classification tasks.
#-------------------------------------------------------------------------------------------------------------------------------------------
print('>>>>> Step 1: Importing libraries and setting up the environment')
import os  # noqa: E402
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings  # noqa: E402
import matplotlib.pylab as plt # type: ignore  # noqa: E402
import numpy as np             # type: ignore  # noqa: E402
import logging                 # type: ignore  # noqa: E402
import tensorflow as tf        # type: ignore  # noqa: E402
import tensorflow_hub as hub   # type: ignore  # noqa: E402
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
#------------------------------------------------------------------------------------------------------------------------------------------
# The following line of code sets the Matplotlib backend to 'agg', which is a non-GUI backend suitable for generating images without 
# displaying them.
# This is particularly useful in environments where you want to save plots to files without needing a display, such as in server-side 
# scripts or automated testing # environments. 
# By using 'agg', you can create high-quality plots and save them as image files (like PNG or JPEG) without rendering them on the screen.
# This is often done to improve performance and avoid issues related to graphical display dependencies, especially in headless environments.
# The 'agg' backend is part of Matplotlib's flexible architecture, allowing users to choose the most appropriate backend
# for their specific use case. In this case, it ensures that the code can run efficiently
# without requiring a graphical user interface, making it suitable for batch processing or automated workflows.
# Note: If you want to display plots interactively in a Jupyter notebook or similar environment,
# you would typically use a different backend, such as 'inline' or 'notebook'.
# However, for this code snippet, 'agg' is used to focus on generating and saving plots without displaying them.
#------------------------------------------------------------------------------------------------------------------------------------------
plt.switch_backend('agg')
#------------------------------------------------------------------------------------------------------------------------------------------
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
#-------------------------------------------------------------------------------------------------------------------------------------------
print('>>>>> Step 2: Loading the pre-trained Inception V1 model from TensorFlow Hub')
model = hub.KerasLayer("https://tfhub.dev/google/imagenet/inception_v1/classification/4", input_shape=(224, 224, 3), trainable=False)
#------------------------------------------------------------------------------------------------------------------------------------------
# The `load_imagenet_labels` function is a utility that downloads and loads the ImageNet class labels, which are essential for interpreting the 
# model's predictions in human-readable format. This function bridges the gap between the model's numerical output (class indices) and meaningful 
# category names that users can understand. # The function begins with `tf.keras.utils.get_file('ImageNetLabels.txt', file_path)`, which is a 
# powerful utility that handles file downloading and caching automatically. If the file doesn't exist locally, Keras will download it from the 
# provided URL and save it with the filename 'ImageNetLabels.txt'. On subsequent calls, it will use the cached local copy, making the function 
# efficient for repeated use. This approach is particularly useful in machine learning workflows where you don't want to re-download large files 
# every time you run your code. # The file reading process uses a context manager with `with open(labels_file) as reader:`, which is a Python best 
# practice that ensures the file is properly closed even if an error occurs during reading. Inside this block, `reader.read()` loads the entire 
# file content into memory as a single string. This is appropriate for the ImageNet labels file since it's relatively small (containing 1,000 
# class names), making the memory usage negligible. The key transformation happens with `f.splitlines()`, which converts the single string 
# containing all labels into a list where each element represents one class name. The `splitlines()` method automatically handles different 
# line ending conventions (Unix, Windows, Mac), making the code robust across different operating systems. Each line in the ImageNet labels file 
# corresponds to one of the 1,000 classes that the model can predict, ordered by class index (0-999).
# Finally, `return np.array(labels)` converts the Python list into a NumPy array, which provides several advantages for this use case. 
# NumPy arrays offer more efficient indexing operations, which is important when you need to quickly look up class names using prediction indices. 
# Additionally, NumPy arrays integrate seamlessly with other parts of the machine learning pipeline, and their indexing behavior is more 
# predictable when dealing with integer indices from model predictions. This design allows you to easily convert a model's prediction 
# (like class index 285) into a readable label (like "Egyptian cat") by simply using `labels[285]`.
# The `imagenet_labels` variable is initialized by calling the `load_imagenet_labels` function with the URL of the ImageNet labels file.
# This file contains the human-readable names for the 1,000 classes that the Inception V1 model can predict. 
# By loading these labels, the code prepares to interpret the model's predictions, allowing users to understand which class 
# corresponds to each prediction index. The labels are stored in a NumPy array, which facilitates efficient indexing and retrieval of 
# class names based on the model's output.
# The URL 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt' points to the location where the 
# ImageNet labels file is hosted, ensuring that the labels are always up-to-date and accessible. 
# This approach is particularly useful in machine learning applications where models are trained on standard datasets like ImageNet, 
# as it provides a consistent way to map model predictions to human-readable labels.
#-------------------------------------------------------------------------------------------------------------------------------------------
print('>>>>> Step 3: Loading ImageNet labels for model predictions')
def load_imagenet_labels(file_path):
  labels_file = tf.keras.utils.get_file('ImageNetLabels.txt', file_path)
  with open(labels_file) as reader:
    f      = reader.read()
    labels = f.splitlines()
  return np.array(labels)
imagenet_labels = load_imagenet_labels('https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
#-------------------------------------------------------------------------------------------------------------------------------------------
# The `read_image` function is designed to read an image file, decode it, and preprocess it for input into a neural network model.
# This function is essential for preparing images in a format that the model can understand, particularly for models like Inception V1, 
# which expect images to be of a specific size and format.
# The function begins by reading the image file using `tf.io.read_file(file_name)`,
# which loads the image data into memory as a byte string. This is a common practice in TensorFlow for handling image files, as it allows 
# for flexible processing of various image formats (JPEG, PNG, etc.). The `file_name` parameter should be a string representing the path 
# to the image file.
# Next, the image byte string is decoded into a tensor using `tf.io.decode_jpeg(image, channels=3)`.
# This step converts the raw byte data into a numerical representation that TensorFlow can work with. 
# The `channels=3` argument specifies that the image # should be decoded as an RGB image with three color channels. 
# If the image is in a different format (like grayscale), this will ensure it is converted to RGB.
# After decoding, the image tensor is converted to a floating-point representation using `tf.image.convert_image_dtype(image, tf.float32)`.
# This step is crucial because many neural network models, including Inception V1,
# expect input images to be in the range [0, 1] rather than the typical [0, 255] range for pixel values. 
# Converting to `tf.float32` ensures that the pixel values are in the correct format for further processing.
# The final step is resizing the image to a fixed size using `tf.image.resize_with_pad(image, target_height=224, target_width=224)`.
# This function resizes the image to 224x224 pixels, which is the standard input size for many convolutional neural networks, including Inception V1.
# The `resize_with_pad` function also pads the image if necessary, ensuring that the aspect ratio is maintained while resizing. 
# This is important because neural networks typically require inputs to have a consistent size, and resizing without padding could distort the image.
# The function returns the preprocessed image tensor, which is now ready to be fed into the neural network model for inference or training.
# In summary, the `read_image` function performs the following steps:
# 1. Reads the image file from the specified path.
# 2. Decodes the image into a tensor with three color channels (RGB).
# 3. Converts the image tensor to a floating-point representation in the range [0, 1].
# 4. Resizes the image to 224x224 pixels with padding to maintain the aspect ratio.
#-------------------------------------------------------------------------------------------------------------------------------------------  
print('>>>>> Step 4: Defining the function to read and preprocess images')
def read_image(file_name):
  image = tf.io.read_file(file_name)
  image = tf.io.decode_jpeg(image, channels=3)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.image.resize_with_pad(image, target_height=224, target_width=224)
  return image
#-------------------------------------------------------------------------------------------------------------------------------------------
# The `img_url` dictionary contains URLs for three different images, each associated with a specific name.
# These images are used to demonstrate the Integrated Gradients technique on various objects, such as a fireboat, a giant panda, and a coyote.
# The URLs point to publicly accessible images hosted on Google Cloud Storage and Wikimedia Commons.
# The `img_paths` dictionary is created using a dictionary comprehension that iterates over the `img_url` items.
# For each (name, url) pair in `img_url`, it uses `tf.keras.utils.get_file` to download the image file from the specified URL and save it locally.
# The `get_file` function handles downloading the file and caching it, so if the file has already been downloaded, it will use the cached
# version instead of downloading it again. The downloaded files are named after the corresponding image names with a `.jpg` extension.
# The `img_name_tensors` dictionary is created by iterating over the `img_paths` items.
# For each (name, img_path) pair, it calls the `read_image` function to read and preprocess the image file located at `img_path`.
# The `read_image` function processes the image by reading it, decoding it, converting it to a floating-point representation, and resizing it to the 
# standard input size of 224x224 pixels, which is suitable for the Inception V1 model.
# The resulting image tensors are stored in the `img_name_tensors` dictionary, where each key is the image name and the value is the 
# corresponding preprocessed image tensor.
# This setup allows for easy access to the preprocessed images by their names, which can be useful for further analysis or visualization.
# The images are displayed using Matplotlib, where each image is shown in a subplot with its corresponding name as the title.
# The `plt.tight_layout()` function is called to adjust the spacing between subplots for better visibility.
# Finally, the figure is saved as 'images/IG-01.jpg' with a resolution of 300 DPI.
# This code snippet effectively demonstrates how to load, preprocess, and visualize images for use in the Integrated Gradients technique
# and provides a clear visual representation of the images being analyzed.
#-------------------------------------------------------------------------------------------------------------------------------------------
print('>>>>> Step 5: Downloading and preprocessing images for Integrated Gradients demonstration')
img_url = {
    'Fireboat'   : 'http://storage.googleapis.com/download.tensorflow.org/example_images/San_Francisco_fireboat_showing_off.jpg',
    'Giant Panda': 'http://storage.googleapis.com/download.tensorflow.org/example_images/Giant_Panda_2.jpeg',
    'coyote'     : 'https://upload.wikimedia.org/wikipedia/commons/c/ce/Canis_latrans_%28Yosemite%2C_2009%29.jpg'
}
img_paths        = {name: tf.keras.utils.get_file(f"{name}.jpg", url) for (name, url) in img_url.items()}
img_name_tensors = {name: read_image(img_path) for (name, img_path) in img_paths.items()}
#--------------------------------------------------------------------------------------------------------------------------------------------
# The following code snippet uses Matplotlib to visualize the preprocessed images stored in the `img_name_tensors` dictionary.
# It creates a figure with a size of 8x8 inches and iterates over the items in `img_name_tensors`.
# For each image tensor, it creates a subplot and displays the image using `ax.imshow(img_tensors)`.
# The title of each subplot is set to the corresponding image name using `ax.set_title(name)`.
# The axis is turned off for each subplot using `ax.axis('off')`, which removes the ticks and labels
# for a cleaner look.
# After all subplots are created, `plt.tight_layout()` is called to adjust the spacing
# between the subplots for better visibility. Finally, the figure is saved as 'images/IG-01.jpg' with a resolution of 300 DPI.
# This code effectively visualizes the images in a grid layout, allowing for easy comparison and analysis of the different images used 
# in the Integrated Gradients technique. The saved image can be used for documentation or presentation purposes.
#--------------------------------------------------------------------------------------------------------------------------------------------
print('>>>>> Step 6: Visualizing preprocessed images')
plt.figure(figsize=(8, 8))
for n, (name, img_tensors) in enumerate(img_name_tensors.items()):
  ax = plt.subplot(1, 3, n+1)
  ax.imshow(img_tensors)
  ax.set_title(name)
  ax.axis('off')
plt.tight_layout()
plt.savefig('images/IG-01.jpg', dpi=300)
#--------------------------------------------------------------------------------------------------------------------------------------------
# The `top_k_predictions` function is designed to make predictions using a pre-trained model and return the top-k predicted class labels 
# along with their probabilities.
# This function is particularly useful for understanding the model's confidence in its predictions and for interpreting the results of image 
# classification tasks.
# The function takes two parameters:
# 1. `img`: A preprocessed image tensor that is ready to be fed into the model.
# 2. `k`: An integer specifying the number of top predictions to return (default is 3).
# Inside the function, the image tensor is expanded to create a batch of size 1 using `tf.expand_dims(img, 0)`. 
# This is necessary because the model expects input in the form of batches, even if there is only one image being processed.
# The model is then called with the image batch, which produces predictions in the form of logits (raw output scores for each class).
# The logits are converted to probabilities using `tf.nn.softmax(predictions, axis=-1)`, which normalizes the scores to a range between 0 and 1, 
# ensuring that they sum to 1 across all classes.
# The `tf.math.top_k` function is used to retrieve the top-k probabilities and their corresponding class indices from the probabilities tensor.
# The `input=probs` argument specifies the input tensor from which to extract the top-k values, and `k=k` indicates the number of 
# top values to return.
# The resulting `top_probs` tensor contains the top-k probabilities, and `top_idxs` contains the indices of the corresponding classes.
# The class labels for the top-k predictions are then retrieved from the `imagenet_labels` array using `imagenet_labels[tuple(top_idxs)]`.
# This step maps the class indices back to their human-readable labels, allowing for easy interpretation
# of the model's predictions.
# Finally, the function returns a tuple containing the top-k class labels and their corresponding probabilities.
# This function is useful for evaluating the model's performance on specific images and for understanding which classes 
# the model is most confident about.
# It can be used in various applications, such as image classification, object detection, and other computer vision tasks 
# where model interpretability is important.
#--------------------------------------------------------------------------------------------------------------------------------------------
print('>>>>> Step 7: Defining the function to get top-k predictions')
def top_k_predictions(img, k=3):
  image_batch         = tf.expand_dims(img, 0)
  predictions         = model(image_batch)
  probs               = tf.nn.softmax(predictions, axis=-1)
  top_probs, top_idxs = tf.math.top_k(input=probs, k=k)
  top_labels          = imagenet_labels[tuple(top_idxs)]
  return top_labels, top_probs[0]
#-------------------------------------------------------------------------------------------------------------------------------------------
# The following code snippet iterates over the `img_name_tensors` dictionary, which contains preprocessed image tensors.
# For each image tensor, it displays the image using Matplotlib, sets the title to the image name in bold font, and turns off the axis 
# for a cleaner look.
# The image is saved as a JPEG file with a resolution of 300 DPI, using the format 'images/IG-02-{name}.jpg', where `{name}` is the image name.
# After displaying the image, it prints the top-k predictions for each image, including the image shape, data type, pixel range, mean, 
# and standard deviation.
# It also prints the top predictions along with their probabilities.
# This code effectively visualizes the images and provides detailed information about the predictions made by the model.
# The saved images can be used for documentation or presentation purposes, while the printed predictions help in understanding 
# the model's performance on each image.
# The `top_k_predictions` function is called for each image tensor to retrieve the top-k predicted class labels and their probabilities.
# The predictions are printed in a formatted manner, showing the class labels and their corresponding probabilities.
#--------------------------------------------------------------------------------------------------------------------------------------------
print('>>>>> Step 8: Displaying images and printing top-k predictions')
print('-------------------------------------')
for (name, img_tensor) in img_name_tensors.items():
  plt.imshow(img_tensor)
  plt.title(name, fontweight='bold')
  plt.axis('off')
  plt.savefig(f'images/IG-02-{name}.jpg', dpi=300)
  print(f'Top {len(imagenet_labels)} predictions for {name}:')
  print('-------------------------------------')
  print(f'Image shape       : {img_tensor.shape}')
  print(f'Image pixel range : {tf.reduce_min(img_tensor):0.2f} to {tf.reduce_max(img_tensor):0.2f}')
  print(f'Image pixel mean  : {tf.reduce_mean(img_tensor):0.2f}')
  print(f'Image pixel stddev: {tf.math.reduce_std(img_tensor):0.2f}')
  pred_label, pred_prob = top_k_predictions(img_tensor)
  for label, prob in zip(pred_label, pred_prob):
    print(f'{label:<18}: {prob:0.1%}')
  print('-------------------------------------')
#-------------------------------------------------------------------------------------------------------------------------------------------
# The following code snippet provides an intuitive understanding of the Integrated Gradients (IG) technique
# by illustrating how gradients can saturate over the model's output function F(x).
# It defines a simple model function `f(x)` that simulates the behavior of a neural network output.
# The function returns the input `x` if it is less than 0.8, and returns 0.8 otherwise, effectively capping the output at 0.8.
# This behavior is visualized using Matplotlib, where the x-axis represents pixel values and the y-axis represents the model's predicted 
# probability for the true class.
# The first subplot shows the function `f(x)` with markers indicating the points where gradients are greater than 0 (indicating 
# that the pixel is important for the prediction)
# and where gradients are equal to 0 (indicating that the pixel is not important for the prediction).
# The second subplot illustrates the intuition behind Integrated Gradients by showing a straight line path from the baseline (0.0) 
# to the input (1.0).
# This path represents the accumulation of gradients along the way, which is a key concept in the IG technique.
# The code also includes annotations to explain the significance of the gradients and the path.
# The x-axis is labeled as 'x - (pixel value)', and the y-axis is labeled as 'F(x) - model true class predicted probability'.
# The figure is saved as 'images/IG-03.jpg' with a resolution of 300 DPI.
# This visualization helps to understand how Integrated Gradients work by showing how gradients can change as the input varies,
# and how the accumulation of gradients along a path can provide insights into the model's predictions.
# The use of a simplified model function and a straight line path makes it easier to grasp the core concepts of IG
# without the complexity of a full neural network.
#-------------------------------------------------------------------------------------------------------------------------------------------
print('>>>>> Step 9: Visualizing the intuition behind Integrated Gradients')
def f(x):
  return tf.where(x < 0.8, x, 0.8)
def interpolated_path(x):
  return tf.zeros_like(x)
x   = tf.linspace(start=0.0, stop=1.0, num=6)
y   = f(x)
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
ax1.annotate('Baseline', xy=(0.0, 0.0), xytext=(0.0, 0.2),  arrowprops=dict(facecolor='black', shrink=0.1))
ax1.annotate('Input'   , xy=(1.0, 0.0), xytext=(0.95, 0.2), arrowprops=dict(facecolor='black', shrink=0.1))
plt.savefig('images/IG-03.jpg', dpi=300)
#-------------------------------------------------------------------------------------------------------------------------------------------
# The following code snippet demonstrates the concept of a baseline image in the context of Integrated Gradients (IG).
# A baseline image serves as a reference point for calculating the contributions of different pixels in an input image to the model's predictions.
# In this case, a baseline image is created as a blank image with all pixel values set to zero, which is a common choice for a baseline.
# The baseline image is defined as a 224x224 pixel image with three color channels (RGB), where each pixel value is initialized to zero.
# This means that the baseline image is completely black, representing the absence of any features or information.
# The code then uses Matplotlib to visualize the baseline image.
# It creates a figure with a size of 5x5 inches, displays the baseline image using `plt.imshow(baseline)`, and sets the title to "Baseline".
# The axis is turned off using `plt.axis('off')` to provide a cleaner view of the image without any ticks or labels.
# Finally, the figure is saved as 'images/IG-04.jpg' with a resolution of 300 DPI.
# This baseline image will be used in subsequent steps to compute the Integrated Gradients for the input images.
# The choice of a black baseline image is a common practice in explainability techniques, as it allows for a clear comparison between the
# baseline and the input image, highlighting the contributions of different pixels to the model's predictions.
#-------------------------------------------------------------------------------------------------------------------------------------------
print('>>>>> Step 10: Defining the baseline image for Integrated Gradients')
baseline = tf.zeros(shape=(224,224,3))
plt.figure(figsize=(5, 5))
plt.imshow(baseline)
plt.title("Baseline")
plt.axis('off')
plt.savefig('images/IG-04.jpg', dpi=300)
#-------------------------------------------------------------------------------------------------------------------------------------------
# The following code snippet demonstrates the process of interpolating images between a baseline image and a target image using TensorFlow.
# This is a crucial step in the Integrated Gradients technique, where we create a series of images that gradually transition from the baseline 
# to the target image.
# The `m_steps` variable defines the number of interpolation steps, which is set to 50 in this case.
# The `alphas` tensor is created using `tf.linspace`, which generates a sequence of evenly spaced values between 0.0 and 1.0.
# This tensor will be used to control the interpolation between the baseline and the target image.
# The `interpolate_images` function takes three parameters: `baseline`, `image`, and `alphas`.
# It performs the interpolation by calculating a series of images that transition from the baseline to the target image.
# Inside the function, `alphas_x` is reshaped to match the dimensions of the images, allowing for broadcasting during the interpolation.
# The baseline image is expanded to create a batch dimension, and the target image is also expanded similarly.
# The difference between the target image and the baseline image is computed as `delta`.
# The interpolated images are then calculated by adding the scaled difference (`alphas_x * delta`) to the baseline image.
# The function returns a tensor containing the interpolated images.
# The `interpolated_images` tensor is created by calling the `interpolate_images` function
# with the baseline image, a specific target image (in this case, 'Fireboat'), and the `alphas` tensor.
# The resulting tensor contains a series of images that gradually transition from the baseline to the target image.
# The code then uses Matplotlib to visualize the interpolated images.
# It creates a figure with a size of 20x20 inches and iterates over the `alphas` tensor, displaying every 10th interpolated image.
# Each image is shown in a subplot with a title indicating the corresponding alpha value.
# The axis is turned off for each subplot to provide a cleaner view of the images.
# Finally, the figure is saved as 'images/IG-05.jpg' with a resolution of 300 DPI.
# This visualization helps to understand how the Integrated Gradients technique works by showing the gradual transition from the baseline
# to the target image, highlighting the contributions of different pixels along the way. 
# The use of interpolation allows for a smooth representation of the pixel contributions, making it easier to analyze the model's behavior.
#-------------------------------------------------------------------------------------------------------------------------------------------
print('>>>>> Step 11: Interpolating images between baseline and target image')
m_steps = 50
alphas  = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)
def interpolate_images(baseline, image, alphas):
  alphas_x   = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
  baseline_x = tf.expand_dims(baseline, axis=0)
  input_x    = tf.expand_dims(image, axis=0)
  delta      = input_x - baseline_x
  images     = baseline_x +  alphas_x * delta
  return images
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
plt.savefig('images/IG-05-1.jpg', dpi=300)
interpolated_images = interpolate_images(baseline=baseline,image=img_name_tensors['Giant Panda'],alphas=alphas)
fig = plt.figure(figsize=(20, 20))
i = 0
for alpha, image in zip(alphas[0::10], interpolated_images[0::10]):
  i += 1
  plt.subplot(1, len(alphas[0::10]), i)
  plt.title(f'alpha: {alpha:.1f}')
  plt.imshow(image)
  plt.axis('off')
plt.tight_layout()
plt.savefig('images/IG-05-2.jpg', dpi=300)
interpolated_images = interpolate_images(baseline=baseline,image=img_name_tensors['coyote'],alphas=alphas)
fig = plt.figure(figsize=(20, 20))
i = 0
for alpha, image in zip(alphas[0::10], interpolated_images[0::10]):
  i += 1
  plt.subplot(1, len(alphas[0::10]), i)
  plt.title(f'alpha: {alpha:.1f}')
  plt.imshow(image)
  plt.axis('off')
plt.tight_layout()
plt.savefig('images/IG-05-3.jpg', dpi=300)
#-------------------------------------------------------------------------------------------------------------------------------------------
# The following code snippet demonstrates how to compute the gradients of the model's output with respect to
# the interpolated images using TensorFlow's GradientTape.
# This is a crucial step in the Integrated Gradients technique, as it allows us to understand how the model's predictions change
# as we move along the interpolated path from the baseline to the target image.
# The `compute_gradients` function takes two parameters: `images`, which is a tensor containing the interpolated images,
# and `target_class_idx`, which is the index of the target class for which we want to compute the gradients.
# Inside the function, a `tf.GradientTape` context is used to record operations for automatic differentiation.
# The `tape.watch(images)` line ensures that the gradients will be computed with respect to the `images` tensor.
# The model is then called with the `images` tensor, producing logits (raw output scores) for each class.
# The probabilities for the target class are computed using `tf.nn.softmax(logits, axis=-1)[:, target_class_idx]`.
# This line applies the softmax function to the logits to convert them into probabilities, and then selects the probability 
# for the specified target class index.
# The gradients of the probabilities with respect to the images are computed using `tape.gradient(probs, images)`.
# This line calculates the gradients of the target class probabilities with respect to the input images, which will be used to understand 
# the contributions of different # pixels to the model's predictions.
# The computed gradients are returned as the output of the function.
# The `path_gradients` tensor is created by calling the `compute_gradients` function
# with the `interpolated_images` tensor and the target class index (in this case, 555, which corresponds to the class "Fireboat").
# The `pred` tensor contains the model's predictions for the interpolated images, and `pred_proba` contains the probabilities for the 
# target class.
# The code then uses Matplotlib to visualize the predicted probabilities and the average pixel gradients over the interpolated path.
# The first subplot shows the predicted probabilities for the target class over the alpha values,
# while the second subplot displays the average pixel gradients normalized over the alpha values.
# The average gradients are computed by taking the mean of the `path_gradients` tensor along the spatial dimensions (height, width, and channels).
# The normalization is done by subtracting the minimum value from each element and dividing by the range (maximum - minimum) 
# of the `average_grads` tensor.
# This ensures that the pixel gradients are scaled to a range between 0 and 1, making it easier to visualize and interpret the results.
# The figure is saved as 'images/IG-06.jpg' with a resolution of 300 DPI.
# This code effectively demonstrates how to compute and visualize the gradients of the model's output with respect to the interpolated images,
# providing insights into how the model's predictions change with respect to the input image pixels.
# The resulting plots help to understand the contributions of different pixels to the model's predictions,
# allowing for a better understanding of the model's behavior and the importance of each pixel in the context of Integrated Gradients.
#-------------------------------------------------------------------------------------------------------------------------------------------
print('>>>>> Step 12: Computing gradients along the interpolated path')
def compute_gradients(images, target_class_idx):
  with tf.GradientTape() as tape:
    tape.watch(images)
    logits = model(images)
    probs  = tf.nn.softmax(logits, axis=-1)[:, target_class_idx]
  return tape.gradient(probs, images)
path_gradients = compute_gradients(images = interpolated_images, target_class_idx = 555)
pred       = model(interpolated_images)
pred_proba = tf.nn.softmax(pred, axis=-1)[:, 555]
plt.figure(figsize=(10, 4))
ax1 = plt.subplot(1, 2, 1)
ax1.plot(alphas, pred_proba)
ax1.set_title('Target class predicted probability over alpha')
ax1.set_ylabel('model p(target class)')
ax1.set_xlabel('alpha')
ax1.set_ylim([0, 1])
ax2 = plt.subplot(1, 2, 2)
#------------------------------------------------------------------------------------------------------------------------------------------
# The following code calculates the average pixel gradients over the interpolated path and normalizes them.
# This is an important step in the Integrated Gradients technique, as it helps to understand how
# the model's predictions change with respect to the input image pixels.
# The `average_grads` tensor is computed by taking the mean of the `path_gradients` tensor along the spatial dimensions (height, width, 
# and channels).
# This results in a tensor that represents the average gradient for each pixel across the entire interpolated path.
# The `tf.reduce_mean` function is used to compute the mean along the specified axes, which are 1, 2, and 3 in this case.
# The `average_grads_norm` tensor is then calculated by normalizing the `average_grads` tensor.
# The normalization is done by subtracting the minimum value from each element and dividing
# by the range (maximum - minimum) of the `average_grads` tensor.
# This normalization ensures that the pixel gradients are scaled to a range between 0 and 1,
# making it easier to visualize and interpret the results.
# The `ax2` subplot is used to plot the normalized average pixel gradients over the alpha values.
# The `ax2.plot` function is called with the `alphas` tensor on the x-axis and the `average_grads_norm` tensor on the y-axis.
# The title of the subplot is set to "Average pixel gradients (normalized) over alpha",
# and the y-axis is labeled as "Average pixel gradients".
# The x-axis is labeled as "alpha", and the y-axis limits are set to [0, 1] to ensure that the plot is properly scaled.
# Finally, `plt.tight_layout()` is called to adjust the layout of the subplots
# for better visibility, and the figure is saved as 'images/IG-06.jpg' with a resolution of 300 DPI.
# This code effectively demonstrates how to compute and visualize the average pixel gradients over the interpolated path in the context of
# Integrated Gradients.
# The resulting plot provides insights into how the model's predictions change with respect to the input image pixels,
# allowing for a better understanding of the model's behavior and the contributions of different pixels to the final prediction.
#-------------------------------------------------------------------------------------------------------------------------------------------
print('>>>>> Step 13: Calculating average pixel gradients over the interpolated path')
average_grads      = tf.reduce_mean(path_gradients, axis=[1, 2, 3])
average_grads_norm = (average_grads-tf.math.reduce_min(average_grads))/(tf.math.reduce_max(average_grads)-tf.reduce_min(average_grads))
ax2.plot(alphas, average_grads_norm)
ax2.set_title('Average pixel gradients (normalized) over alpha')
ax2.set_ylabel('Average pixel gradients')
ax2.set_xlabel('alpha')
ax2.set_ylim([0, 1])
plt.tight_layout()
plt.savefig('images/IG-06.jpg', dpi=300)
#-------------------------------------------------------------------------------------------------------------------------------------------
# The following code snippet defines a function `integral_approximation` that computes the average
# of the gradients along the interpolated path in the context of Integrated Gradients.
# This function takes a tensor of gradients as input and calculates the average gradient by averaging
# the gradients at each step along the path.
# The `gradients` parameter is expected to be a tensor containing the gradients computed over the interpolated path.
# The function first computes the average of the gradients at each step by taking the mean of the gradients
# at adjacent steps. 
# This is done using the expression `(gradients[:-1] + gradients[1:]) / tf.constant(2.0)`, which averages the
# gradients at each pair of adjacent steps.
# The resulting `grads` tensor contains the average gradients at each step along the path.
# The function then computes the integrated gradients by taking the mean of the averaged gradients across all steps.
# This is done using `tf.math.reduce_mean(grads, axis=0)`, which reduces the tensor along the first dimension (the steps) to obtain a 
# single tensor representing the average gradient.
# The final result, `integrated_gradients`, is returned as the output of the function.
# This function is crucial in the Integrated Gradients technique, as it provides a way to approximate
# the integral of the gradients along the path from the baseline to the input image.
# By averaging the gradients at each step, it captures the contributions of different pixels to the model's predictions,
# allowing for a better understanding of the model's behavior and the importance of each pixel.
# The computed integrated gradients can be used for visualization and interpretation of the model's predictions,
# helping to identify which pixels are most influential in the model's decision-making process.
#-------------------------------------------------------------------------------------------------------------------------------------------
print('>>>>> Step 14: Defining the function to compute integrated gradients')
def integral_approximation(gradients):
  grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
  integrated_gradients = tf.math.reduce_mean(grads, axis=0)
  return integrated_gradients
#-------------------------------------------------------------------------------------------------------------------------------------------
# The following code snippet demonstrates the use of the `integral_approximation` function to
# compute the integrated gradients for a given set of path gradients.
# The `path_gradients` tensor is expected to contain the gradients computed over the interpolated
# path from the baseline to the input image.
# The `integral_approximation` function is called with the `path_gradients` tensor as an argument,
# and it returns the integrated gradients as a tensor.
# This tensor represents the average gradients along the path, capturing the contributions of different pixels to the
# model's predictions.
# The computed integrated gradients can be used for further analysis and visualization, allowing for a better understanding
# of the model's behavior and the importance of each pixel in the input image.
# The `integral_approximation` function is defined earlier in the code, and it performs the following steps:
# 1. It averages the gradients at each step along the path by taking the mean of the gradients at adjacent steps.
# 2. It computes the integrated gradients by taking the mean of the averaged gradients across all steps.
# 3. It returns the integrated gradients as the output.
# This code effectively demonstrates how to compute the integrated gradients using the `integral_approximation` function,
# providing a crucial step in the Integrated Gradients technique for interpreting model predictions.
# The integrated gradients can be visualized and analyzed to understand the model's decision-making process
# and the contributions of different pixels to the final prediction.
#-------------------------------------------------------------------------------------------------------------------------------------------
print('>>>>> Step 15: Computing integrated gradients from path gradients')
ig = integral_approximation(gradients=path_gradients)
#-------------------------------------------------------------------------------------------------------------------------------------------
# The following code snippet defines a function `integrated_gradients` that computes the Integrated Gradients (IG) for a given image.
# This function is a key part of the Integrated Gradients technique, which is used to interpret
# the contributions of different pixels in an image to the model's predictions.
# The function takes four parameters:
# 1. `baseline`: The baseline image, which serves as a reference point for the computation of IG.
# 2. `image`: The target image for which the IG is to be computed.
# 3. `target_class_idx`: The index of the target class for which the IG is to be computed.
# 4. `m_steps`: The number of interpolation steps (default is 50).
# 5. `batch_size`: The size of the batches for processing the images (default is 32).
# The function first creates a tensor `alphas` that contains evenly spaced values between 0.0 and 1.0,
# representing the interpolation steps. The number of steps is determined by the `m_steps` parameter.
# It then initializes an empty list `gradient_batches` to store the computed gradients for each batch of images.
# The function iterates over the `alphas` tensor in batches defined by the `batch_size`.
# For each batch, it calls the `one_batch` function, which computes the gradients for the interpolated images between the baseline 
# and the target image.
# The `one_batch` function takes the baseline image, target image, alpha values for the batch, and the target class index as inputs.
# It first interpolates the images using the `interpolate_images` function, which generates a series of images that transition from 
# the baseline to the target image based on the alpha values.
# It then computes the gradients of the model's predictions with respect to these interpolated images using the `compute_gradients` function.
# The computed gradients for each batch are appended to the `gradient_batches` list.
# After processing all batches, the function concatenates the computed gradients along the first axis to obtain a single tensor `total_gradients` 
# containing all the gradients.
# The average gradients are then computed using the `integral_approximation` function, which averages the gradients at each step along the path.
# Finally, the Integrated Gradients are calculated by multiplying the difference between the target image and the baseline image with 
# the average gradients.
# The function returns the computed Integrated Gradients tensor, which represents the contributions of different pixels in the target 
# image to the model's predictions.
# This tensor can be used for visualization and interpretation of the model's behavior, helping to identify
# which pixels are most influential in the model's decision-making process.
#-------------------------------------------------------------------------------------------------------------------------------------------
print('>>>>> Step 16: Defining the function to compute integrated gradients')
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
#--------------------------------------------------------------------------------------------------------------------------------------------
# The `one_batch` function is defined to compute the gradients for a batch of interpolated images.
# This function is called within the `integrated_gradients` function to process the images in smaller batches,
# which is useful for handling large datasets or when the number of interpolation steps is high.
# The function takes four parameters:
# 1. `baseline`: The baseline image, which serves as a reference point for the computation of Integrated Gradients.
# 2. `image`: The target image for which the gradients are to be computed.
# 3. `alpha_batch`: A batch of alpha values that represent the interpolation steps between the baseline and the target image.
# 4. `target_class_idx`: The index of the target class for which the gradients are to be computed.
# The function first calls the `interpolate_images` function to generate a batch of interpolated images.
# This function takes the baseline image, target image, and the alpha values for the batch as inputs.
# It computes a series of images that transition from the baseline to the target image based on the alpha values.
# The interpolated images are stored in the `interpolated_path_input_batch` tensor.
# Next, the function computes the gradients of the model's predictions with respect to these interpolated images.
# This is done using the `compute_gradients` function, which takes the interpolated images and the target class index as inputs.
# The `compute_gradients` function uses TensorFlow's `GradientTape` to record the operations and compute the gradients of the model's predictions
# with respect to the interpolated images.
# The computed gradients for the batch are stored in the `gradient_batch` tensor.
# Finally, the function returns the `gradient_batch` tensor, which contains the gradients for the interpolated images in the batch.
# This function is crucial for the Integrated Gradients technique, as it allows for efficient computation of gradients
# over a large number of interpolation steps by processing the images in smaller batches.
# By breaking down the computation into manageable chunks, it helps to reduce memory usage
# and improve the overall performance of the Integrated Gradients algorithm.
#--------------------------------------------------------------------------------------------------------------------------------------------
print('>>>>> Step 17: Defining the function to compute gradients for one batch of interpolated images')
def one_batch(baseline, image, alpha_batch, target_class_idx):
    interpolated_path_input_batch = interpolate_images(baseline=baseline, image=image, alphas=alpha_batch)
    gradient_batch                = compute_gradients(images = interpolated_path_input_batch,target_class_idx=target_class_idx)
    return gradient_batch
#-------------------------------------------------------------------------------------------------------------------------------------------
# The following code snippet demonstrates the use of the `integrated_gradients` function to compute the Integrated Gradients for a specific image.
# The `integrated_gradients` function is called with the following parameters:
# - `baseline`: The baseline image, which serves as a reference point for the computation of Integrated Gradients.
# - `image`: The target image for which the Integrated Gradients are to be computed.
# - `target_class_idx`: The index of the target class for which the Integrated Gradients are to be computed.
# - `m_steps`: The number of interpolation steps (default is 50).
# The function returns the Integrated Gradients for the specified image, which is stored in the `ig_attributions` tensor.
# The shape of the `ig_attributions` tensor is printed to verify the dimensions of the computed attributions.
# The code also prints a separator line for better readability in the output.
# This code is useful for understanding how the Integrated Gradients technique works by computing the pixel-wise contributions of the target image to the model's predictions.
# The resulting `ig_attributions` tensor can be used for visualization and interpretation of the model's behavior,
# helping to identify which pixels are most influential in the model's decision-making process.
# The `integrated_gradients` function is defined earlier in the code, and it performs the following steps:
# 1. Computes the gradients of the model's predictions with respect to the input images.
# 2. Averages the gradients over the interpolation steps.
# 3. Scales the averaged gradients by the difference between the input image and the baseline image.
# 4. Returns the scaled gradients as the Integrated Gradients attributions.
# The computed Integrated Gradients can be visualized and analyzed to understand the model's decision-making process
# and the contributions of different pixels to the final prediction.
#-------------------------------------------------------------------------------------------------------------------------------------------- 
print('>>>>> Step 18: Computing Integrated Gradients for specific images')
ig_attributions = integrated_gradients(baseline=baseline, image=img_name_tensors['Fireboat'], target_class_idx=555, m_steps=240)
print('Attributions shape Fireboat   :', ig_attributions.shape)
ig_attributions = integrated_gradients(baseline=baseline, image=img_name_tensors['Giant Panda'], target_class_idx=555, m_steps=240)
print('Attributions shape Giant Panda:', ig_attributions.shape)
ig_attributions = integrated_gradients(baseline=baseline, image=img_name_tensors['coyote'], target_class_idx=555, m_steps=240)
print('Attributions shape Coyote     :', ig_attributions.shape)
#-------------------------------------------------------------------------------------------------------------------------------------------
# The following code snippet defines a function `plot_img_attributions` that visualizes the Integrated Gradients (IG) attributions for a given image.
# This function is useful for interpreting the contributions of different pixels in the image to the model's predictions.
# The function takes the following parameters:
# - `name`: The name of the image, which will be used in the title and filename of the plot.
# - `baseline`: The baseline image, which serves as a reference point for the computation of IG.
# - `image`: The target image for which the IG attributions are to be computed.
# - `target_class_idx`: The index of the target class for which the IG attributions are to be computed.
# - `m_steps`: The number of interpolation steps (default is 50).
# - `cmap`: The colormap to be used for visualizing the attribution mask (default is None).
# - `overlay_alpha`: The alpha value for overlaying the attribution mask on the original image (default is 0.4).
# The function first computes the Integrated Gradients attributions using the `integrated_gradients` function,
# which is defined earlier in the code.
# The attributions are computed by passing the baseline image, target image, target class index, and the number of interpolation steps as arguments.
# The attributions are then reduced to a mask by summing the absolute values of the attributions along the color channels.
# The attribution mask is stored in the `attribution_mask` tensor.
# The function then creates a Matplotlib figure with a 2x2 grid of subplots to visualize the results.
# The first subplot displays the baseline image with the title "Baseline image".
# The second subplot shows the original image with the title "Original image".
# The third subplot visualizes the attribution mask using the specified colormap (`cmap`), with the title "Attribution mask".
# The fourth subplot overlays the attribution mask on the original image with the specified alpha value (`overlay_alpha`),
# and the title "Overlay".
# Each subplot has its axis turned off for a cleaner look.
# The figure is then saved as a JPEG file with a resolution of 300 DPI, using the format 'images/IG-07-{name}.jpg',
# where `{name}` is the name of the image.
# Finally, the function returns the created figure object.
# This function is crucial for visualizing the Integrated Gradients attributions, allowing for a better understanding of how different pixels
# contribute to the model's predictions.
# The resulting plots can be used for documentation, analysis, or presentation purposes, helping to interpret the model's behavior and the
# importance of each pixel in the input image.
# The use of color maps and overlays enhances the visualization, making it easier to identify the regions of the image that have the most 
# significant impact on the model's predictions.
#--------------------------------------------------------------------------------------------------------------------------------------------
print('>>>>> Step 19: Defining the function to visualize Integrated Gradients attributions')
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
  return fig
#-------------------------------------------------------------------------------------------------------------------------------------------
# The following code snippet demonstrates how to use the `plot_img_attributions` function to visualize
# the Integrated Gradients (IG) attributions for specific images.
# The function is called with the name of the image, the baseline image, the target image for which the IG attributions are to be computed,
# the target class index, the number of interpolation steps, the colormap to be used for visualizing the attribution mask, and the alpha value for overlaying the attribution mask on the original image.
# The first call to `plot_img_attributions` visualizes the IG attributions for the image 'Fireboat'.
# The baseline image is passed as `baseline`, the target image is `img_name_tensors['Fireboat']`, the target class index is set to 555,
# and the number of interpolation steps is set to 240.
# The colormap used is `plt.cm.inferno`, and the overlay alpha value is set to 0.4.
# The second call visualizes the IG attributions for the image 'Giant Panda'.
# The baseline image is the same, the target image is `img_name_tensors['Giant Panda']`, the target class index is set to 389,
# and the number of interpolation steps is set to 55.
# The colormap used is `plt.cm.viridis`, and the overlay alpha value is set to 0.5.
# The third call visualizes the IG attributions for the image 'coyote'.
# The baseline image is the same, the target image is `img_name_tensors['coyote']`, the target class index is also set to 389,
# and the number of interpolation steps is set to 55.
# The colormap used is again `plt.cm.viridis`, and the overlay alpha value is set to 0.5.
# Each call to `plot_img_attributions` generates a figure that visualizes the baseline image, the original image,
# the attribution mask, and the overlay of the attribution mask on the original image.
# The resulting figures are saved as JPEG files with appropriate names based on the image being processed.
# This code effectively demonstrates how to visualize the Integrated Gradients attributions for different images,
# allowing for a better understanding of how different pixels contribute to the model's predictions.
#-------------------------------------------------------------------------------------------------------------------------------------------
print('>>>>> Step 20: Visualizing Integrated Gradients attributions for specific images')
_ = plot_img_attributions(name='Fireboat',    baseline=baseline, image=img_name_tensors['Fireboat'],   target_class_idx=555, m_steps=240, cmap=plt.cm.inferno, overlay_alpha=0.4)
_ = plot_img_attributions(name='Giant Panda', image=img_name_tensors['Giant Panda'],baseline=baseline, target_class_idx=389, m_steps=55,  cmap=plt.cm.viridis, overlay_alpha=0.5)
_ = plot_img_attributions(name='coyote',      image=img_name_tensors['coyote']     ,baseline=baseline, target_class_idx=389, m_steps=55,  cmap=plt.cm.viridis, overlay_alpha=0.5)
#-------------------------------------------------------------------------------------------------------------------------------------------